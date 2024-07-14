from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import os
import numpy as np
import json
import argparse
import time

os.environ['HF_DATASETS_OFFLINE'] = '1'

      

parser = argparse.ArgumentParser()
parser.add_argument('--json_file', type=str, default='configs/config.json')
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()

with open(args.json_file, 'r') as f:
    config = json.load(f)
config['TrainingArgs']['local_rank'] = args.local_rank

if config['TrainingArgs']['local_rank'] == 0:
    print(config)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

raw_datasets = load_dataset("data/imdb.py", cache_dir='data/cache')
tokenizer = AutoTokenizer.from_pretrained("./data/cache/tokenizer")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

model = AutoModelForSequenceClassification.from_pretrained(
    "./data/cache/bert-base-cased",
    num_labels=2
    )

metric = load_metric("./data/accuracy.py")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(**config['TrainingArgs'])
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=full_train_dataset,
    eval_dataset=full_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
