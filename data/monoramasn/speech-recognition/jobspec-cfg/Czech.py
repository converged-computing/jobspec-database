import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import evaluate
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List, Union
from datasets import DatasetDict, Audio, load_from_disk, concatenate_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

from datasets import load_dataset, load_metric

cs_voxpopuli_dataset = load_dataset("facebook/voxpopuli", "cs", split=['train', 'test'])

dataset = cs_voxpopuli_dataset.remove_columns(['audio_id', 'language', 'raw_text', 'gender', 'speaker_id', 'is_gold_transcript', 'accent'])


#dataset

gradient_checkpointing = True
freeze_feature_encoder = False
freeze_encoder = False

do_normalize_eval = True
do_lower_case = False
do_remove_punctuation = False
normalizer = BasicTextNormalizer()


model_checkpoint= "openai/whisper-large"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_checkpoint)
tokenizer = WhisperTokenizer.from_pretrained(model_checkpoint, language="Czech", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_checkpoint, language="Czech", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_checkpoint)

if model.config.decoder_start_token_id is None:
    raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

if freeze_feature_encoder:
    model.freeze_feature_encoder()

if freeze_encoder:
    model.freeze_encoder()
    model.model.encoder.gradient_checkpointing = False


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

if gradient_checkpointing:
    model.config.use_cache = False

model_checkpoint_name = model_checkpoint.split("/")[-1]
repo_name = f"{model_checkpoint_name}-demo-colab"

def prepare_dataset(batch):
    # Load and resample audio data to the expected sampling rate
    audio = batch["audio"]
    input_features = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    #input_features = input_features.reshape(-1, 80, 3000)

    # Ensure the last dimension of input_features is 3000
    if input_features.shape[-1] < 3000:
        padding = torch.zeros(3000 - input_features.shape[-1])
        input_features = torch.cat([input_features, padding], dim=0)

    batch["input_features"] = input_features

    # Compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    # Optional pre-processing steps
    transcription = batch["normalized_text"]
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()

    # Encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription, padding="max_length", max_length=max_label_length).input_ids

    return batch


max_label_length = model.config.max_length
min_input_length = 0.0
max_input_length = 30.0
def is_in_length_range(length, labels):
    return min_input_length < length < max_input_length and 0 < len(labels) < max_label_length

    # Apply preprocessing and ensure 'labels' key is added
dataset = dataset.map(prepare_dataset, batch_size=32)

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

print('DATASET PREPARATION COMPLETED')

import numpy as np
#metric=evaluate.load_metric("wer")

from datasets import load_metric
#metric = load_metric("wer")
wer_metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

    model

import torch
import torch.nn as nn
import torch.nn.functional as F


#parser = argparse.ArgumentParser()
#parser.add_argument("--language", type=str, default="")
#parser.add_argument("--model_size", type=str, default="")
#args = parser.parse_args()
#language = args.language

# Attention mechanism
class WhisperAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, k, v, q):
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self.q_proj(q)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))
        attn_probs = F.softmax(attn_scores, dim=-1)

        context = torch.matmul(attn_probs, v)
        return self.out_proj(context)

# Encoder layer
class WhisperEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.self_attn = WhisperAttention(d_model)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.self_attn_layer_norm(x + attn_output)

        fc_output = self.fc2(self.activation_fn(self.fc1(x)))
        x = self.final_layer_norm(x + fc_output)
        return x

# Encoder
class WhisperEncoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, d_ff, max_len):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([WhisperEncoderLayer(d_model, d_ff) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # Transpose to match the (batch_size, seq_len, features) format
        x = x.transpose(1, 2)  # now shape: (batch_size, seq_len, d_model)

        # Adjusting the position_ids based on the actual sequence length after convolution
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device)
        position_embeddings = self.embed_positions(position_ids)  # now shape: (seq_len, d_model)

        x += position_embeddings

        for layer in self.layers:
            x = layer(x)

        x = self.layer_norm(x)
        return x

# Decoder layer
class WhisperDecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.self_attn = WhisperAttention(d_model)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attn = WhisperAttention(d_model)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output):
        self_attn_output = self.self_attn(x, x, x)
        x = self.self_attn_layer_norm(x + self_attn_output)

        enc_attn_output = self.encoder_attn(encoder_output, encoder_output, x)
        x = self.encoder_attn_layer_norm(x + enc_attn_output)

        fc_output = self.fc2(self.activation_fn(self.fc1(x)))
        x = self.final_layer_norm(x + fc_output)
        return x
# Decoder
class WhisperDecoder(nn.Module):
    def __init__(self, d_model, n_layers, d_ff, max_len, vocab_size):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model, padding_idx=50257)
        self.embed_positions = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([WhisperDecoderLayer(d_model, d_ff) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output):
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device)
        x = self.embed_tokens(x) + self.embed_positions(position_ids)

        for layer in self.layers:
            x = layer(x, encoder_output)

        x = self.layer_norm(x)
        return x

# Complete model
class WhisperForfinetune(nn.Module):
    def __init__(self, input_dim=80, encoder_d_model=512, encoder_n_layers=6, encoder_d_ff=2048, max_len=1500, vocab_size=51865):
        super().__init__()
        self.encoder = WhisperEncoder(input_dim, encoder_d_model, encoder_n_layers, encoder_d_ff, max_len)
        self.decoder = WhisperDecoder(encoder_d_model, encoder_n_layers, encoder_d_ff, max_len, vocab_size)
        self.proj_out = nn.Linear(encoder_d_model, vocab_size, bias=False)

    def forward(self, input_features, labels):
        encoder_output = self.encoder(input_features)
        decoder_output = self.decoder(labels, encoder_output)
        logits = self.proj_out(decoder_output)

        outputs = {'logits': logits}
        if labels is not None:
          loss_fn = nn.CrossEntropyLoss()
          # Reshape labels and logits to compute loss, adjust dimensions as necessary
          loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
          outputs['loss'] = loss

        return outputs
        
# Instantiate the model
model1 = WhisperForfinetune()

# Example input (dummy data)
#input_features = torch.randn(1, 80, 3000)  # Batch size x Input dimension x Sequence length
#labels = torch.randint(0, 51865, (1, 100))  # Example target token sequence

# Forward pass
#output = model1(input_features, labels)

from transformers import TrainingArguments

training_args = Seq2SeqTrainingArguments(
  output_dir=repo_name,
  group_by_length=True,
  per_device_train_batch_size=8,
  evaluation_strategy="steps",
  num_train_epochs=50,
  #fp16=False,
  gradient_checkpointing=False,
  save_steps=50,
  eval_steps=50,
  logging_steps=50,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=2,
  push_to_hub=True,
)

from transformers import Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model1,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
trainer.evaluate()
