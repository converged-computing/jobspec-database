import os
import torch
import numpy as np
import evaluate
import transformers
from pathlib import Path
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
)
from argparse import ArgumentParser
from datasets import load_from_disk

parser = ArgumentParser()
parser.add_argument("--datapath", default=f'{os.environ["DATASET_STORE"]}/prepared_hf')
parser.add_argument("--optim", default='SGD', type=str)
parser.add_argument("--dd_rho", default=0.5, type=float)
parser.add_argument("--asam_eta", default=0.01, type=float)
parser.add_argument("--samson_norm", default="inf", type=str)
args = parser.parse_args()

MODEL_NAME = "google/vit-base-patch16-224-in21k"

assert args.optim in ['ASAM', 'SAMSON', 'SAM', 'SGD']
optimizer_name = f'{args.optim}'
if args.optim in ['ASAM', 'SAMSON', 'SAM']:
    optimizer_name += f'_rho{args.dd_rho}'
if args.optim in ['ASAM', 'SAMSON']:
    optimizer_name += f'_eta{args.asam_eta}'
if args.optim in ['SAMSON']:
    optimizer_name += f'_norm{args.samson_norm}'

###
# Dataset
###
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
assert Path(f'{args.datapath}/imagenet1k.hfdatasets').is_dir()
ds = load_from_disk(f'{args.datapath}/imagenet1k.hfdatasets')

def transform(example_batch):
    inputs = processor([x.convert("RGB") for x in example_batch["image"]], return_tensors="pt")
    inputs["label"] = example_batch["label"]
    return inputs


prepared_ds = ds.with_transform(transform)


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }


metric = evaluate.load("accuracy")


def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )


###
# Models
###
labels = prepared_ds["train"].features["label"].names
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
)


# Optimizers
opt = torch.optim.SGD(model.parameters(), lr=0.06, momentum=0.9, weight_decay=0.)
sched = transformers.get_cosine_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=20000)
t = Trainer

# Minimizers
from optimizer_ascending import SAM, ASAM, SAMSON
from trainer_ascending import ASAMTrainer
if args.optim == 'SAM':
    minimizer = SAM
elif args.optim == 'ASAM':
    minimizer = ASAM
elif args.optim == 'SAMSON':
    minimizer = SAMSON

if args.optim == 'SGD':
    pass
else:
    t = ASAMTrainer
    opt.minimizer = minimizer(opt, model, rho=args.dd_rho, eta=args.asam_eta, norm=args.samson_norm)

###
# Finetuning args https://arxiv.org/pdf/2010.11929.pdf
# SGD momentum 0.9 no wd
# CosineLR decay
# batch size 512
# no wd -> default of trainer
# - grad clipping global norm = 1
# - resolution 384
# max steps = 20000
# - base lr in {0.003, 0.01, 0.03, 0.06}
###
training_args = TrainingArguments(
    output_dir=f"./{optimizer_name}",
    #per_device_train_batch_size=128,
    #gradient_accumulation_steps=4,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    evaluation_strategy="steps",
    fp16=True,
    save_steps=2000,
    eval_steps=2000,
    logging_steps=100,
    #learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="tensorboard",
    load_best_model_at_end=True,
    max_steps=20000,
)


###
# Finetune
###
trainer = t(
    model=model,
    args=training_args,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
    tokenizer=processor,
    optimizers=(opt, sched),
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepared_ds["validation"])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
