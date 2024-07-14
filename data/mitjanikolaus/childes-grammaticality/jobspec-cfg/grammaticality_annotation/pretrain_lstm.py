import argparse
import math
import os.path
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset

import pandas as pd

from torch import nn
import pytorch_lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from grammaticality_annotation.data import load_childes_data, train_val_split
from grammaticality_annotation.tokenizer import (train_tokenizer, TOKEN_PAD, TOKENIZERS_DIR,
                                                 TOKEN_SPEAKER_CHILD, TOKEN_SPEAKER_CAREGIVER)
from utils import PROJECT_ROOT_DIR

DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data", "manual_annotation", "all")

LM_DATA = os.path.expanduser("~/data/childes_grammaticality/sentences.csv")

BATCH_SIZE = 100

MAX_SEQ_LENGTH = 200    # (number of characters)

MAX_EPOCHS = 10

LSTM_HIDDEN_DIM = 512

LSTM_TOKENIZER_PATH = os.path.join(TOKENIZERS_DIR, "tokenizer_lstm.json")

NUM_VAL_SENTENCES = 10000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CHILDESLMDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]["speaker_code"] + self.data.iloc[idx]["transcript_clean"]
        transcript = self.data.iloc[idx]["transcript_file"]
        idx += 1
        while idx < len(self.data) and transcript == self.data.iloc[idx]["transcript_file"]:
            if len(sentence + self.data.iloc[idx]["speaker_code"] + self.data.iloc[idx]["transcript_clean"]) < MAX_SEQ_LENGTH:
                sentence = sentence + self.data.iloc[idx]["speaker_code"] + self.data.iloc[idx]["transcript_clean"]
            else:
                break
            idx += 1

        sentence = sentence[:MAX_SEQ_LENGTH]

        return sentence


class CHILDESLMDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, tokenizer, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer

        print("loading data.. ", end="")
        data = pd.read_csv(LM_DATA, index_col=0)
        print("done.")

        print("Creating train and val splits")
        data_train, data_val = train_val_split(data, NUM_VAL_SENTENCES)

        self.train_ds = CHILDESLMDataset(data_train)
        self.val_ds = CHILDESLMDataset(data_val)

        self.num_workers = num_workers

    def tokenize_batch(self, batch):
        encodings = self.tokenizer.batch_encode_plus(batch, padding=True, return_tensors="pt")

        encodings.data["labels"] = encodings["input_ids"][:, 1:]

        return encodings

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=self.tokenize_batch, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, collate_fn=self.tokenize_batch, num_workers=self.num_workers)


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, num_labels=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if num_layers > 1:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                                dropout=dropout_rate, batch_first=True)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.fc_classification = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, hidden=None, attention_mask=None, token_type_ids=None):
        if not hidden:
            hidden = self.init_hidden(input_ids.shape[0])
        embedding = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).cpu().numpy()
        packed_input = pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.lstm(packed_input, hidden)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        logits = self.fc(output)

        return {"logits": logits, "hidden": hidden}

    def forward_classification(self, input_ids, hidden=None, attention_mask=None, token_type_ids=None):
        if not hidden:
            hidden = self.init_hidden(input_ids.shape[0])
        embedding = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).cpu().numpy()
        packed_input = pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input, hidden)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        batch_size = input_ids.shape[0]
        output_last_ts = output[range(batch_size), input_sizes - 1]
        logits = self.fc_classification(output_last_ts)

        return {"logits": logits, "hidden": (hidden, cell)}

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell


class CHILDESGrammarLSTM(LightningModule):
    def __init__(
            self,
            pad_token_id,
            vocab_size,
            tokenizer=None,
            embedding_dim: int = LSTM_HIDDEN_DIM,
            hidden_dim: int = LSTM_HIDDEN_DIM,
            num_layers: int = 1,
            dropout_rate: float = 0.1,
            learning_rate: float = 0.001,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            num_labels = 3,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])

        self.tokenizer = tokenizer

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id)

        self.hidden_dim = hidden_dim

        self.vocab_size = vocab_size
        self.model = LSTM(self.vocab_size, embedding_dim,  hidden_dim, num_layers, dropout_rate, num_labels)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        output = self(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
        )
        labels = batch["labels"]
        logits = output["logits"][:, :-1]
        loss = self.loss_fct(logits.reshape(-1, self.vocab_size), labels.reshape(-1))

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        self.log("train_loss", loss.mean().item(), prog_bar=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
        )
        labels = batch["labels"]
        logits = output["logits"][:, :-1]
        val_loss = self.loss_fct(logits.reshape(-1, self.vocab_size), labels.reshape(-1))

        return {"val_loss": val_loss, "logits": logits}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        print("\n\n")
        print(self.generate(TOKEN_SPEAKER_CHILD, max_seq_len=20, temperature=0.3))
        print(self.generate(TOKEN_SPEAKER_CAREGIVER, max_seq_len=20, temperature=0.3))

        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        return [optimizer]

    def generate(self, prompt, max_seq_len, temperature, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.eval()
        encoding = self.tokenizer.encode_plus(prompt, return_tensors="pt").to(device)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        with torch.no_grad():
            hidden = None
            for i in range(max_seq_len):
                output = self.model(input_ids, attention_mask=attention_mask, hidden=hidden)
                logits = output["logits"]
                hidden = output["hidden"]
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                prediction = torch.multinomial(probs, num_samples=1)

                if prediction.item() == self.tokenizer.eos_token_id:
                    break

                input_ids = torch.cat([input_ids, prediction], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.tensor(1, device=input_ids.device).reshape(1, 1)], dim=1)

        decoded = self.tokenizer.decode(input_ids[0].cpu().numpy())
        self.train()
        return decoded


class LSTMSequenceClassification(CHILDESGrammarLSTM):
    def __init__(
            self,
            pad_token_id: int,
            vocab_size: int,
            tokenizer = None,
            embedding_dim: int = LSTM_HIDDEN_DIM,
            hidden_dim: int = LSTM_HIDDEN_DIM,
            num_layers: int = 1,
            dropout_rate: float = 0.1,
            learning_rate: float = 0.001,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            num_labels: int = 3,
            **kwargs,
    ):
        super().__init__(
                         tokenizer=tokenizer,
                         embedding_dim=embedding_dim,
                         hidden_dim=hidden_dim,
                         num_layers=num_layers,
                         dropout_rate=dropout_rate,
                         learning_rate=learning_rate,
                         adam_epsilon=adam_epsilon,
                         warmup_steps=warmup_steps,
                         weight_decay=weight_decay,
                         pad_token_id=pad_token_id,
                         vocab_size=vocab_size,
                         num_labels=num_labels,
                         )

        self.num_labels = num_labels

    def freeze_base_weights(self):
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc_classification.weight.requires_grad = True
        self.model.fc_classification.bias.requires_grad = True

    def forward(self, **inputs):
        return self.model.forward_classification(**inputs)


def prepare_lm_data():
    print("Preparing data...")
    os.makedirs(os.path.dirname(LM_DATA), exist_ok=True)
    data = load_childes_data(DATA_DIR, exclude_test_data=True)
    data = data[["transcript_file", "transcript_clean", "speaker_code"]]
    data.to_csv(LM_DATA)


def train(args):
    if not os.path.isfile(LM_DATA):
        prepare_lm_data()
    if not os.path.isfile(LSTM_TOKENIZER_PATH):
        train_tokenizer(LSTM_TOKENIZER_PATH, LM_DATA, add_eos_token=False)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=LSTM_TOKENIZER_PATH)
    tokenizer.add_special_tokens({'pad_token': TOKEN_PAD})

    data_module = CHILDESLMDataModule(BATCH_SIZE, tokenizer, num_workers=args.num_workers)

    model = CHILDESGrammarLSTM(tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id,
                               vocab_size=tokenizer.vocab_size, num_layers=args.num_layers,
                               learning_rate=args.learning_rate)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_last=True,
                                            filename="{epoch:02d}-{val_loss:.2f}")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=True, mode="min",
                                        min_delta=0.01, stopping_threshold=0.0)

    tb_logger = TensorBoardLogger(name="logs_pretrain_lstm", save_dir=os.path.curdir)
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        devices=1 if torch.cuda.is_available() else None,
        accelerator="auto",
        val_check_interval=1000,
        callbacks=[checkpoint_callback, early_stop_callback],
        reload_dataloaders_every_n_epochs=1,
        logger=tb_logger,
    )

    print("\n\n\nInitial validation:")
    initial_eval = trainer.validate(model, data_module)
    print(f"Perplexity: {math.exp(initial_eval[0]['val_loss']):.2f}")

    trainer.fit(model, datamodule=data_module)

    final_eval = trainer.validate(model, data_module)
    print(f"Perplexity: {math.exp(final_eval[0]['val_loss']):.2f}")


def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--num-workers",
        type=int,
        default=8,
    )

    argparser.add_argument(
        "--num-layers",
        type=int,
        default=1,
    )

    argparser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
