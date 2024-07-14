import argparse
import time
import logging
from datetime import datetime

import datasets
import evaluate
from transformers import pipeline
import numpy as np
from yaml import safe_load

from inference_new_model import process_text


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Training configuration file.')
    # parser.add_argument('--results_csv', required=True, help='Results CSV file.')
    args = parser.parse_args()
    return args


def load_config(config_path):
    with open(config_path, 'r') as file:
        return safe_load(file)


def log_msg(msg: str):
    print(msg)
    logging.info(msg)


def create_pipeline(model_path):
    return pipeline("ner", model=model_path, aggregation_strategy="simple")


def prepare_datasets(config: dict):
    raw_datasets = {key: datasets.load_from_disk(value["path"]) for (key, value) in config["datasets"].items()}
    label_names = ['O', 'B-p', 'I-p', 'B-i', 'I-i', 'B-g', 'I-g', 'B-t', 'I-t', 'B-o', 'I-o']

    if "test" not in config["datasets"]:
        config["datasets"]["test"] = {
            "name": "Combined Test Dataset",
            "desc": "A combination of various splits from multiple datasets.",
            "path": "path/to/combined_dataset"  # Tento údaj je ilustrativní
        }

    concat_datasets = datasets.DatasetDict({
        "test": datasets.concatenate_datasets(
            [dataset[split] for dataset in raw_datasets.values() for split in dataset if
             split in ['test', 'validation']]
        )
    })

    return label_names, concat_datasets


def evaluate_predictions(predictions, references, metric_evaluator):
    # Vyhodnocení predikcí
    return metric_evaluator.compute(predictions=predictions, references=references)


def extract_tags_from_prediction(prediction):
    # Předpokládáme, že 'prediction' je seznam stringů, kde každý string obsahuje jednu větu ve formátu "slovo tag"
    # Rozdělíme každou větu na slova, a pak extrahujeme pouze tagy
    lines = prediction.split('\n')
    tags = [line.split()[1] if len(line.split()) > 1 else 'O' for line in lines if line.strip()]
    return tags


def main():
    start_time = time.monotonic()
    args = parse_arguments()
    with open(args.config, 'r') as config_file:
        config = safe_load(config_file)
    model1 = create_pipeline(config['models']['model1']['path'])
    model2 = create_pipeline(config['models']['model2']['path'])
    model3 = create_pipeline(config['models']['model3']['path'])

    label_names, test_dataset = prepare_datasets(config)

    metric_evaluator = evaluate.load("seqeval")
    tag_keys = ['p', 'i', 'g', 't', 'o']
    cumulative_results = {
        'overall_accuracy': [],
        'overall_f1': [],
        'overall_precision': [],
        'overall_recall': [],
        **{tag: {'f1': [], 'precision': [], 'recall': []} for tag in tag_keys}
    }

    test_dataset = test_dataset['test']

    sentences = []
    all_references = []
    i = 0

    for example in test_dataset:
        text = " ".join(example['tokens'])
        sentences.append(text)
        references = [label_names[tag_idx] for tag_idx in example['ner_tags']]
        all_references.append(references)
        i += 1
        if i > 100:
            break

    # Provedení hromadné predikce pro všechny shromážděné věty
    prediction_outputs = process_text(sentences, model1, model2, model3)

    all_predictions = [extract_tags_from_prediction(output) for output in prediction_outputs]

    all_predictions = [sublist for sublist in all_predictions if sublist]

    # Hromadné vyhodnocení všech predikcí
    results = evaluate_predictions(all_predictions, all_references, metric_evaluator)

    print(f"Final Evaluation Results: {results}")

    end_time = time.monotonic()
    log_msg("Elapsed script time: {}\n".format(datetime.timedelta(seconds=end_time - start_time)))


if __name__ == "__main__":
    main()
