from config import *
from pynvml import *
import re
import pandas as pd

from string import digits, punctuation
from datasets import Dataset
from data_cleaning import DataObject

from transformers import AutoTokenizer, AutoModel, pipeline
import torch


# setting up data
# note that the sample csvs can contain multiple replications for a study
train_df = pd.read_csv(working_path + "Data/training_sample.csv")
train_texts = pd.read_csv(working_path + "Data/dataset_training.csv")

# note that training set holds 388 papers, but only 348 unique dois (because multiple attempts at same study)
# which means training set actually = 348 data points
# we don't need to store the text multiple times ...
print("unique training data, n=", len(train_texts['doi'].unique()))

####
data_object = DataObject(train_df, train_texts)
data = data_object.modify_data()

Xs = data['full_text']
ys = data['label']

print("count", len(Xs))
print("replicated", 100.0 * (ys.where(ys == "yes").count()/len(Xs)), "%")

data.to_csv(working_path + "/Data/full_training_data.csv", index=False)


# Modeling

model_name = "domenicrosati/deberta-v3-large-dapt-tapt-scientific-papers-pubmed-finetuned-DAGPap22"

####

tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=128)

def yesNoToInt(row):
  y = row.label
  return int(y == "yes")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

######

data["labels"] = data.apply(yesNoToInt, axis=1)
data["labels"] = pd.to_numeric(data["labels"])

dataset = Dataset.from_pandas(data).shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.15)
dataset = dataset.rename_column("full_text", "text")
dataset = dataset.remove_columns(["id", "doi", "label"])

tokenized_papers = dataset.map(preprocess_function, batched=True)

####

from transformers import DataCollatorWithPadding
import evaluate
import numpy as np

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

id2label = {0: "no", 1: "yes"}
label2id = {"no": 0, "yes": 1}

###

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="deberta-replication",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=100,
    eval_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_papers["train"],
    eval_dataset=tokenized_papers["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

save = True
if save:
  trainer.save_model("our_model")
