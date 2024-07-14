import os
import torch
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from collections import defaultdict
from model import load_pretrained_model
import matplotlib.pyplot as plt
import json
import tempfile
import shutil
from composer.models import write_huggingface_pretrained_from_composer_checkpoint


def icl_tokenize(tokenizer, context_window=4096):
    def icl_tokenize_inner(examples):
        context_indices = tokenizer.encode(examples["context"])
        continuation_indices = tokenizer.encode(examples["continuation"])

        return {
            "context_indices": context_indices,
            "continuation_indices": continuation_indices,
        }

    return icl_tokenize_inner


def icl_collate_fn(tokenizer):
    def icl_collate_fn_inner(examples):
        context_indices = [example["context_indices"] for example in examples]
        continuation_indices = [example["continuation_indices"] for example in examples]

        context_indices = torch.nn.utils.rnn.pad_sequence(
            torch.tensor(context_indices, dtype=torch.long),
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )

        continuation_indices = torch.nn.utils.rnn.pad_sequence(
            torch.tensor(continuation_indices, dtype=torch.long),
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )

        return {
            "context_indices": context_indices,
            "continuation_indices": continuation_indices,
        }

    return icl_collate_fn_inner


def inference(
    checkpoint_path,
    jsonl_file,
    output_dir,
    context_window=4096,
    batch_size=16,
    use_gpu=False,
):
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    # Create a temporary directory for the Hugging Face model
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert the .pt checkpoint to a Hugging Face model
        hf_model_path = os.path.join(temp_dir, "hf_model")
        write_huggingface_pretrained_from_composer_checkpoint(
            checkpoint_path, hf_model_path
        )

        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            hf_model_path, max_len=context_window, padding_side="right", truncation=True
        )

        dataset = load_dataset(
            "json",
            data_files={"eval": jsonl_file},
            cache_dir=os.path.join(os.path.dirname(jsonl_file), ".cache"),
        ).map(icl_tokenize(tokenizer, context_window=context_window))["eval"]

        dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=icl_collate_fn(tokenizer),
            pin_memory=True,
            num_workers=cpu_count(),
        )

        model = load_pretrained_model(hf_model_path, device)

        space_token_id = tokenizer.convert_tokens_to_ids(" ")
        star_token_id = tokenizer.convert_tokens_to_ids("*")

        results = []

        for data in tqdm(dataloader):
            input_ids = data["context_indices"].to(device)
            continuation_ids = data["continuation_indices"].to(device)

            generated_token_ids = []

            while True:
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                    generated_token_id = torch.argmax(outputs.logits[0, -1]).item()
                    generated_token_ids.append(generated_token_id)

                    if (
                        generated_token_id != space_token_id
                        and generated_token_id != star_token_id
                    ):
                        break

                    input_ids = torch.cat(
                        (input_ids, torch.tensor([[generated_token_id]]).to(device)),
                        dim=1,
                    )

            context_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            context = tokenizer.convert_tokens_to_string(context_tokens)

            label_tokens = tokenizer.convert_ids_to_tokens(continuation_ids[0].tolist())
            label = tokenizer.convert_tokens_to_string(label_tokens)

            prediction_tokens = tokenizer.convert_ids_to_tokens(generated_token_ids)
            prediction = tokenizer.convert_tokens_to_string(prediction_tokens)

            results.append(
                {
                    "context": context,
                    "context_length": len(context_tokens),
                    "label": label,
                    "prediction": prediction,
                }
            )

    # create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    last_checkpoint_folder = os.path.basename(checkpoint_path)
    output_dir = os.path.join(
        output_dir, f"FINAL-{context_window}", last_checkpoint_folder
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save results to a JSONL file
    jsonl_file_name = os.path.basename(jsonl_file.replace(".jsonl", ""))
    jsonl_file = os.path.join(
        output_dir,
        f"inference_results_{jsonl_file_name}.jsonl",
    )

    with open(jsonl_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    return jsonl_file


def compute_metrics(jsonl_file, output_dir, context_window):
    with open(jsonl_file, "r") as f:
        results = [json.loads(line) for line in f]

    stop_token_counts = defaultdict(int)
    generated_star_counts = defaultdict(int)
    correct_predictions = 0
    total_predictions = 0
    star_differences = []
    stats = {}

    correct_predictions_fits = 0
    total_predictions_fits = 0
    correct_predictions_exceeds = 0
    total_predictions_exceeds = 0

    for result in results:
        label_stars = result["label"].count("*")
        prediction_stars = result["prediction"].count("*")

        generated_star_counts[prediction_stars] += 1

        if prediction_stars == label_stars:
            correct_predictions += 1
            if result["context_length"] <= context_window:
                correct_predictions_fits += 1
            else:
                correct_predictions_exceeds += 1
        total_predictions += 1
        if result["context_length"] <= context_window:
            total_predictions_fits += 1
        else:
            total_predictions_exceeds += 1

        star_difference = abs(prediction_stars - label_stars)
        star_differences.append(star_difference)

    # Compute generated star length statistics
    for star_length, count in generated_star_counts.items():
        percentage = count / total_predictions * 100
        print(f"Generated {star_length} stars: {percentage:.2f}% of the time")
        stats[f"generated_star_counts/{star_length}"] = percentage

    # Compute accuracy and mean star difference
    accuracy = correct_predictions / total_predictions
    mean_star_difference = sum(star_differences) / len(star_differences)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean Star Difference: {mean_star_difference:.2f}")

    stats["accuracy"] = accuracy
    stats["mean_star_difference"] = mean_star_difference

    # Compute accuracy for context rows that fit within context window and those that don't
    accuracy_fits = (
        correct_predictions_fits / total_predictions_fits
        if total_predictions_fits > 0
        else 0
    )
    accuracy_exceeds = (
        correct_predictions_exceeds / total_predictions_exceeds
        if total_predictions_exceeds > 0
        else None
    )

    print(f"Accuracy (fits within context window): {accuracy_fits:.4f}")
    print(f"Accuracy (exceeds context window): {accuracy_exceeds:.4f}")

    stats["accuracy_fits"] = accuracy_fits
    stats["accuracy_exceeds"] = accuracy_exceeds

    output_dir = os.path.dirname(jsonl_file)
    os.makedirs(output_dir, exist_ok=True)

    # Save evaluation results to a CSV file
    csv_file = os.path.join(
        output_dir,
        f"eval_results_{os.path.basename(jsonl_file.replace('.jsonl', ''))}_{context_window}.csv",
    )

    with open(csv_file, "w") as f:
        f.write("metric,value\n")
        for metric, value in stats.items():
            if value is None:
                f.write(f"{metric},N/A\n")
            else:
                f.write(f"{metric},{value}\n")

    # Create matplotlib figure for generated star counts
    fig_generated_stars = plt.figure(figsize=(7, 5))
    plt.bar(generated_star_counts.keys(), generated_star_counts.values())
    plt.xlabel("Number of Stars")
    plt.ylabel("Count")
    plt.title("Generated Star Counts")

    # save figure to output directory
    fig_generated_stars_file = os.path.join(
        output_dir,
        f"generated_star_counts_{os.path.basename(jsonl_file.replace('.jsonl', ''))}_{context_window}.png",
    )

    fig_generated_stars.savefig(fig_generated_stars_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="The path to the checkpoint file to evaluate.",
    )
    parser.add_argument(
        "--jsonl_file",
        type=str,
        required=True,
        help="The JSONL file to use for evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory to save the evaluation results.",
    )
    parser.add_argument(
        "--context_window",
        type=int,
        default=4096,
        help="The context window size for tokenization.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="The batch size for evaluation."
    )
    parser.add_argument(
        "--use_gpu", action="store_true", help="Whether to use GPU for inference."
    )

    args = parser.parse_args()

    jsonl_file = inference(
        args.checkpoint_path,
        args.jsonl_file,
        args.output_dir,
        args.context_window,
        args.batch_size,
        args.use_gpu,
    )

    compute_metrics(jsonl_file, args.output_dir, args.context_window)
