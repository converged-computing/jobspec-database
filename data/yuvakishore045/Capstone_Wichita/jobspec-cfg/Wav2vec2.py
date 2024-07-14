import os
import IPython
import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
import torchaudio
import ssl
import pythainlp
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML
import re
import soundfile
from pythainlp import tokenize
ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pythainlp.tokenize import word_tokenize
from datasets import load_dataset
import json
import numpy as np

#repo='Checkpoints'
data=load_dataset("librispeech_asr", "clean", "train.100")
speech = data.remove_columns(['speaker_id', 'chapter_id', 'id'])

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch

speech = speech.map(remove_special_characters)
def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocabs = speech.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=speech.column_names["train.100"])
vocab_list = list(set(vocabs["train.100"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
# repo='Checkpoints'
# tokenizer.push_to_hub(repo_name)

from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

speech = speech.map(prepare_dataset, remove_columns=speech.column_names["train.100"], num_proc=4)
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
from datasets import load_metric
wer_metric = load_metric("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


from transformers import Wav2Vec2ForCTC

# model = Wav2Vec2ForCTC.from_pretrained(
#     "facebook/wav2vec2-base", 
#     ctc_loss_reduction="mean", 
#     pad_token_id=processor.tokenizer.pad_token_id)
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h",ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id,)
model.freeze_feature_extractor()
from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="check3",
  group_by_length=True,
  per_device_train_batch_size=8,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=False,
  gradient_checkpointing=True, 
  save_steps=1000,
  eval_steps=1000,
  logging_steps=1000,
  learning_rate=1e-3,
  weight_decay=0.005,
  warmup_steps=2000,
  save_total_limit=2)

from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=speech["train.100"],
    eval_dataset=speech["validation"],
    tokenizer=processor.feature_extractor,
)
trainer.train()
# trainer.push_to_hub(repo)

def map_to_result(batch):
  with torch.no_grad():
    input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  batch["pred_str"] = processor.batch_decode(pred_ids)[0]
  batch["text"] = processor.decode(batch["labels"], group_tokens=False)
  
  return batch

results = speech["test"].map(map_to_result, remove_columns=speech["test"].column_names)
print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))
