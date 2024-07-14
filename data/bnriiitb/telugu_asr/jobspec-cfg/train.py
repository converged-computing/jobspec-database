import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset,concatenate_datasets, Audio

from huggingface_hub import login
login("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")
# screen 
# nvidia-smi  check the available GPUs
# export CUDA_VISIBLE_DEVICES=15 use the appropriate GPU
# python train.py |& tee console.log
# screen -ls
# screen -r $screen_running
# du -sh .
# rm -rf .cache/

# all PATHs related constants
BASE_PATH = "/raid/cs20mds14030/telugu_asr/data"

INDIC_SUPREB_DATASETS = ["indic_superb/clean_train","indic_superb/clean_valid",
                         "indic_superb/clean_test_known","indic_superb/clean_test_unknown"]

OPENSLR_DATASETS = ["open_slr/te_in_female","open_slr/te_in_male"]

ULCA_DATASETS = ["ulca/Chai_Bisket_Stories_16-08-2021_14-17",
                 "ulca/Telangana_Sahitya_Akademi_16-08-2021_14-40"]

# ULCA_DATASETS = ["ulca/BBC_News_Telugu_17-08-2021_00-57",
#                  "ulca/Chai_Bisket_Stories_16-08-2021_14-17",
#                  "ulca/Telangana_Sahitya_Akademi_16-08-2021_14-40"]


MUCS_DATASETS = ["mucs/te-in-Test/Audios","mucs/te-in-Train/Audios"]

# Load the dataset from metadata.csv files
DATASETS = INDIC_SUPREB_DATASETS+ULCA_DATASETS+OPENSLR_DATASETS+MUCS_DATASETS

def create_dataset_from_metadata(dataset_name):
    ds = load_dataset('csv', data_files=f"{BASE_PATH}/{dataset_name}/metadata.csv")
    # add the complete file name
    ds = ds.map(lambda x: {'file_name':BASE_PATH+"/"+dataset_name+"/"+x["file_name"]})
    return ds['train']

def load_datasets_from_metadata(DATASETS):
    print(DATASETS[0])
    ds = create_dataset_from_metadata(DATASETS[0])
    print(ds)
    for dataset_name in DATASETS[1:]:
        temp_ds = create_dataset_from_metadata(dataset_name)
        ds = concatenate_datasets([ds,temp_ds])
        print(ds)
    return ds

print(f'Avaialble datasets --> {DATASETS}')

print('##### Loading the datasets #####')
ds = load_datasets_from_metadata(DATASETS)
print('##### Successfully loaded the datasets #####')

# normalize the sampling rate to 16k Hz
ds = ds.cast_column("file_name",Audio(sampling_rate=16000))

# rename the column file_name to audio
ds = ds.rename_column("file_name", "audio")

audio_duration = round((np.sum(ds['duration'])/60)/60)
print({f"The dataset has {ds.num_rows} rows worth {audio_duration} hours of data"})

# perform train test split. As the dataset is large limiting the test_size to just 10%
train_test_dataset = ds.train_test_split(test_size=0.10)
print(train_test_dataset)
print("Performed train test split")

# compute trainign and testing set audio duration
train_audio_duration = round((np.sum(train_test_dataset['train']['duration'])/60)/60)
test_audio_duration = round((np.sum(train_test_dataset['test']['duration'])/60)/60)

print('Training Dataset Details')
print(train_test_dataset['train'])
print(f"{train_test_dataset['train'].num_rows} samples --> {train_audio_duration} hours of data")
print("\n")

print('Testing Dataset Details')
print(train_test_dataset['test'])
print(f"{train_test_dataset['test'].num_rows} samples --> {train_audio_duration} hours of data")

train_test_dataset = train_test_dataset.remove_columns(['duration'])

model_output_dir = f"./whisper-small-te-{train_audio_duration}h"

print(f'model_output_dir --> {model_output_dir}')

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Telugu", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Telugu", task="transcribe")

print("####### Loaded WhisperFeatureExtractor, WhisperTokenizer and WhisperProcessor successfully #########")

def prepare_dataset(batch):
    audio = batch["audio"]
    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

print("###### Preparing dataset started ########")
train_test_dataset = train_test_dataset.map(prepare_dataset, remove_columns=train_test_dataset.column_names["train"], num_proc=16)
print("####### Preparing dataset completed ########")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


import evaluate
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


from transformers import WhisperForConditionalGeneration
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
print("####### Loaded the Pre-Trained Checkpoint ########")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=model_output_dir,  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=1000,
    max_steps=30000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=5000,
    eval_steps=5000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)
print("####### Defined the Training Configuration #########")

from transformers import Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_test_dataset["train"],
    eval_dataset=train_test_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

print("######## Finetuning Started #########")
# fine tune the model using the check point
trainer.train()
print("######## Finetuning Completed #########")

kwargs = {
    "dataset_tags": "INDIC SUPERB, MUCS, OPENSLR",
    "dataset": "Crowed sourced dataset",
    "dataset_args": "config: te, split: test",
    "language": "te",
    "model_name": "Whisper Small Telugu - Naga Budigam",
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
}
print("######## Pushing the finetuned model to Hugging Face Hub Started #########")
trainer.push_to_hub(**kwargs)
print("######## Pushing the finetuned model to Hugging Face Hub Completed #########")