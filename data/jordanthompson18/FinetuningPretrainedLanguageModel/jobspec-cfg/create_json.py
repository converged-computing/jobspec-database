import jsonlines
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import numpy as np
import os
import random

filename = 'wrong_predictions.jsonl'

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True)

model_checkpoint = "deberta-v3-base-finetuned-imdb/checkpoint-12500/"
batch_size=2

dataset = load_dataset('imdb')
metric = load_metric("accuracy")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)



encoded_dataset = dataset.map(preprocess_function, batched=True)

num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

metric_name = "accuracy"
model_name = model_checkpoint.split("/")[-1]

task = 'imdb'
args = TrainingArguments(
    os.path.join("./", f"{model_name}-finetuned-{task}"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

preds = trainer.predict(encoded_dataset["test"])
pred_labels = []
for logit in preds[0]:
    pred_labels.append(logit.argmax())

print(pred_labels)
print(preds[1])

inds = []
for i in range(len(pred_labels)):
    if pred_labels[i] != preds[1][i]:
        inds.append(i)

inds = random.choices(inds, k=10)
filename = "wrong_labels.jsonl"
output_items = []
for ind in inds:
    output_items.append({"review" : encoded_dataset["test"][ind]["text"], "label" : str(preds[1][ind]), "predicted" : str(pred_labels[ind])})

with jsonlines.open(filename, mode='w') as writer:
    for item in output_items:
        writer.write(item)
    writer.close()


