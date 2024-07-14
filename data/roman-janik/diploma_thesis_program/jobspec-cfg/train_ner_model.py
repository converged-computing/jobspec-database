# Author: Roman Janík
# Script for training baseline model from RobeCzech on CNEC 2.0 CoNNL and CHNEC 1.0 datasets.
#
# Partially taken over from Hugging Face Course Chapter 7 Token classification:
# https://huggingface.co/course/en/chapter7/2?fw=pt
#

import argparse
import csv
import datetime
import logging
import os
import time

import datasets
import transformers
import torch
import evaluate
import numpy as np
import pandas as pd

from accelerate import Accelerator
from tqdm.auto import tqdm
from yaml import safe_load
from torch.utils.tensorboard import SummaryWriter


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Training configuration file.')
    parser.add_argument('--results_csv', required=True, help='Results CSV file.')
    args = parser.parse_args()
    return args


def log_msg(msg: str):
    print(msg)
    logging.info(msg)


def log_summary(exp_name: str, config: dict):
    log_msg("{:<24}{}\n{:<24}{}".format(
        "Name:", exp_name.removeprefix("exp_configs_ner/").removesuffix(".yaml"), "Description:", config["desc"]))
    ct = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg("{:<24}{}\n{:<24}{}\n{:<24}{}\n".format(
        "Start time:", ct, "Model:", config["model"]["name"],
        "Datasets:", [dts["name"] for dts in config["datasets"].values()]))

    cf_t = config["training"]
    log_msg("Parameters:\n{:<24}{}\n{:<24}{}\n{:<24}{}".format(
        "Num train epochs:", cf_t["num_train_epochs"], "Batch size:", cf_t["batch_size"],
        "Val batch size:", cf_t["val_batch_size"]))
    log_msg("{:<24}{}\n{:<24}{}\n{:<24}{}\n{:<24}{}".format(
        "Learning rate:", cf_t["optimizer"]["learning_rate"], "Weight decay:", cf_t["optimizer"]["weight_decay"],
        "Lr scheduler:",
        cf_t["lr_scheduler"]["name"], "Warmup ratio:", cf_t["lr_scheduler"]["warmup_ratio"]))


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def prepare_datasets(config: dict):
    raw_datasets = {key: datasets.load_from_disk(value["path"]) for (key, value) in config["datasets"].items()}
    label_names = ['O', 'B-p', 'I-p', 'B-i', 'I-i', 'B-g', 'I-g', 'B-t', 'I-t', 'B-o', 'I-o']

    # concatenate datasets
    concat_dataset_train = datasets.concatenate_datasets(
        [raw_dataset["train"] for raw_dataset in raw_datasets.values()]
    )
    concat_dataset_validation = datasets.concatenate_datasets(
        [raw_dataset["validation"] for raw_dataset in raw_datasets.values()]
    )

    # initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config["model"]["path"], add_prefix_space=True)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, padding=True, is_split_into_words=True
        )
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    t_concat_dataset_train = concat_dataset_train.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=concat_dataset_train.column_names,
    )

    t_concat_dataset_validation = concat_dataset_validation.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=concat_dataset_train.column_names,
    )

    raw_datasets_test = {dataset_name: raw_dataset["test"] for (dataset_name, raw_dataset) in raw_datasets.items()}

    return tokenizer, label_names, raw_datasets_test, {
        "train": t_concat_dataset_train,
        "validation": t_concat_dataset_validation
    }


# noinspection PyArgumentList
def main():
    start_time = time.monotonic()
    output_dir = "../results"
    model_dir = "../results/model"
    log_dir = "../results/logs"
    args = parse_arguments()

    # Load config file
    with open(args.config, 'r') as config_file:
        config = safe_load(config_file)

    # Start logging, print experiment configuration
    logging.basicConfig(filename=os.path.join(output_dir, "experiment_results.txt"), level=logging.INFO,
                        encoding='utf-8', format='%(message)s')
    log_msg("Experiment summary:\n")
    log_summary(args.config, config)
    log_msg("-" * 80 + "\n")

    # Init tensorboard writer
    writer = SummaryWriter(log_dir)

    tokenizer, label_names, test_datasets, tokenized_datasets = prepare_datasets(config)
    data_collator = transformers.DataCollatorForTokenClassification(tokenizer=tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=config["training"]["batch_size"],
    )

    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_datasets["validation"], collate_fn=data_collator, batch_size=config["training"]["val_batch_size"]
    )

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}

    model = transformers.AutoModelForTokenClassification.from_pretrained(
        config["model"]["path"],
        id2label=id2label,
        label2id=label2id,
    )

    metric = evaluate.load("seqeval")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[lb] for lb in label if lb != -100] for label in labels]
        true_predictions = [
            [label_names[pr] for (pr, lb) in zip(prediction, label) if lb != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    cf_optimizer = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cf_optimizer["learning_rate"],
                                  betas=(cf_optimizer["beta1"], cf_optimizer["beta2"]),
                                  weight_decay=cf_optimizer["weight_decay"])

    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_train_epochs = config["training"]["num_train_epochs"]
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = transformers.get_scheduler(
        config["training"]["lr_scheduler"]["name"],
        optimizer=optimizer,
        num_warmup_steps=int(config["training"]["lr_scheduler"]["warmup_ratio"] * num_training_steps),
        num_training_steps=num_training_steps,
    )

    def postprocess(predictions, labels):
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[lb] for lb in label if lb != -100] for label in labels]
        true_predictions = [
            [label_names[pr] for (pr, lb) in zip(prediction, label) if lb != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_labels, true_predictions

    progress_bar = tqdm(range(num_training_steps))
    step = 0

    # Training loop
    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for i, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            if i % 100 == 99:
                writer.add_scalar("Loss/train", loss, epoch)
            accelerator.backward(loss)

            writer.add_scalar("Learning_rate/train", lr_scheduler.get_last_lr()[0], step)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            step += 1

        # Evaluation
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            # Necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)

            true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
            metric.add_batch(predictions=true_predictions, references=true_labels)

        results = metric.compute()
        print(
            f"\nepoch {epoch}:",
            {
                key: "{:.6f}".format(results[f"overall_{key}"])
                for key in ["f1", "accuracy", "precision", "recall"]
            },
        )
        writer.add_scalars("Metrics/train",
                           {"f1": results["overall_f1"], "accuracy": results["overall_accuracy"],
                            "precision": results["overall_precision"], "recall": results["overall_recall"]
                            }, epoch)

        # Save
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(model_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(model_dir)

    writer.flush()
    time.sleep(3)

    # Log last validation results
    log_msg("-" * 80 + "\nExperiment results:\n")
    log_msg("Validation set evaluation:")
    last_val_results = {
        key: "{:.6f}".format(results[key])
        for key in ["overall_f1", "overall_accuracy", "overall_precision", "overall_recall"]
    }
    log_msg(f"Last epoch {epoch}:\n{last_val_results}\n")

    # Test set evaluation
    log_msg("Test set evaluation:")
    task_evaluator = evaluate.evaluator("token-classification")
    # test_model = transformers.AutoModelForTokenClassification.from_pretrained(
    #     os.path.join(output_dir, "model")
    # )

    test_results = {}
    for (dataset_name, test_dataset) in test_datasets.items():
        test_result = task_evaluator.compute(model_or_pipeline=unwrapped_model, data=test_dataset,
                                             tokenizer=tokenizer, metric="seqeval")
        test_results[dataset_name] = test_result
        test_result_df = pd.DataFrame(test_result).loc["number"]
        log_msg("{}:\n{}\n".format(config["datasets"][dataset_name]["name"],
                                   test_result_df[
                                       ["overall_f1", "overall_accuracy", "overall_precision", "overall_recall"]]))

    # Log to CSV results file
    exp_name = os.path.splitext(os.path.basename(args.config))[0]
    dtss = "_".join(config["datasets"].keys())
    test_cnec_f1 = "{:.6f}".format(test_results["cnec"]["overall_f1"]) if "cnec" in test_results else None
    test_chnec_f1 = "{:.6f}".format(test_results["chnec"]["overall_f1"]) if "chnec" in test_results else None
    test_poner_f1 = "{:.6f}".format(test_results["poner"]["overall_f1"]) if "poner" in test_results else None
    try:
        with open(args.results_csv, encoding="utf-8") as f:
            f.read()
    except FileNotFoundError:
        with open(args.results_csv, "w", encoding="utf-8") as exp_f:
            exp_res_wr = csv.writer(exp_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            exp_res_wr.writerow(["exp_name", "model_name", "datasets", "num_epochs",
                                 "val_f1", "cnec_test_f1", "chnec_test_f1", "poner_test_f1"])
            exp_res_wr.writerow([exp_name, config["model"]["name"], dtss,
                                 config["training"]["num_train_epochs"], "{:.6f}".format(results["overall_f1"]),
                                 test_cnec_f1, test_chnec_f1, test_poner_f1])
    else:
        with open(args.results_csv, "a", encoding="utf-8") as exp_f:
            exp_res_wr = csv.writer(exp_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            exp_res_wr.writerow([exp_name, config["model"]["name"], dtss,
                                 config["training"]["num_train_epochs"], "{:.6f}".format(results["overall_f1"]),
                                 test_cnec_f1, test_chnec_f1, test_poner_f1])

    end_time = time.monotonic()
    log_msg("Elapsed script time: {}\n".format(datetime.timedelta(seconds=end_time - start_time)))


if __name__ == "__main__":
    main()
