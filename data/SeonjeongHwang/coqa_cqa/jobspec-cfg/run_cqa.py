import os
import sys
import random
import json
import tqdm
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np

from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup 
from tool.data_process import *

MIN_FLOAT = -1e30

import argparse

parser = argparse.ArgumentParser(description="CQA")

### Arguments for Traning
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--learning-rate", type=float)
parser.add_argument("--warmup-prop", type=float)

### Directories
parser.add_argument("--output-dir", type=str)
parser.add_argument("--result-dir", type=str)

### Arguments for Dataset
parser.add_argument("--num-turn", type=int, default=3)
parser.add_argument("--max-seq-length", type=int, default=512)
parser.add_argument("--max-history-length", type=int, default=128)
parser.add_argument("--doc-stride", type=int, default=192)
parser.add_argument("--model-name", type=str, default="bert-cased-large")

args = parser.parse_args()

train_data = f"data/coqa/coqa-train-v1.0.json"
train_feature = f"data/coqa/train_{args.num_turn}_features.pkl"
valid_data = f"data/coqa/coqa-dev-v1.0.json"
valid_feature = f"data/coqa/dev_{args.num_turn}_features.pkl"

exp_dir = os.path.join(args.output_dir, args.result_dir)
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(exp_dir, exist_ok=True)

model_file=exp_dir+"/model/model.pth"
tokenizer_dir=exp_dir+"/tokenizer"
os.makedirs(exp_dir+"/model", exist_ok=True)
os.makedirs(tokenizer_dir, exist_ok=True)

config = exp_dir+"/config.json"
config_items = {"model_name": args.model_name,
                "max_seq_length": args.max_seq_length,
                "max_history_length": args.max_history_length,
                "doc_stride": args.doc_stride,
                "num_turn": args.num_turn}

with open(config, "w") as f:
    json.dump(config_items, f, indent=1)

model_name = args.model_name
max_seq_length = args.max_seq_length
max_history_length = args.max_history_length
doc_stride = args.doc_stride
num_turn = args.num_turn
    
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 2022
seed_everything(seed)

class Dataset(Dataset):
    def __init__(self, data_file, feature_file, tokenizer, mode):
        if os.path.exists(feature_file):
            print(f"Loading {mode} features from {feature_file}...")
            with open(feature_file, "rb") as f:
                self.features = pickle.load(f)
        else:
            print(f"Generating {mode} examples...")
            examples = read_example(input_file=data_file, is_training=True, num_turn=num_turn)
            print(f"Generating {mode} features...")
            self.features = convert_examples_to_features(examples=examples, 
                                                         tokenizer=tokenizer, 
                                                         max_seq_length=max_seq_length,
                                                         max_history_length=max_history_length,
                                                         doc_stride=doc_stride,
                                                         is_training=True)
            print(f"Save the features to {feature_file}...")
            with open(feature_file, "wb") as f:
                pickle.dump(self.features, f, pickle.HIGHEST_PROTOCOL)
        
        self.input_ids = self.features["input_ids"]
        self.attention_mask = self.features["attention_mask"]
        self.segment_ids = self.features["segment_ids"]
        self.start_position = self.features["start_position"]
        self.end_position = self.features["end_position"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx])
        attention_mask = torch.tensor(self.attention_mask[idx])
        segment_ids = torch.tensor(self.segment_ids[idx])
        start_position = torch.tensor(self.start_position[idx])
        end_position = torch.tensor(self.end_position[idx])
        
        return input_ids, attention_mask, segment_ids, start_position, end_position

class CQA(nn.Module):
    def __init__(self, bert_model_name, tokenizer):
        super().__init__()        
        self.BertEncoder = BertModel.from_pretrained(bert_model_name)
        self.BertEncoder.resize_token_embeddings(len(tokenizer))
        
        ##### CODE #####
        
    def forward(self, input_ids, segment_ids, attention_mask):
        bert_output = self.BertEncoder(input_ids=input_ids, 
                                       attention_mask=attention_mask, 
                                       token_type_ids=segment_ids).last_hidden_state
        
        ##### CODE #####

    
def fit(model, train_dataset, val_dataset, device, epochs=2, batch_size=12, warmup_prop=0, lr=3e-5):
    progress_bar = tqdm.tqdm
    model = model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))
    num_training_steps = epochs * len(train_loader)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    loss_fct = nn.CrossEntropyLoss(reduction='mean').to(device) # or sum
        
    dev_loss = float("inf")
    for epoch in range(epochs):
        print(f"*** Epoch{epoch} ***")
        train_pbar = progress_bar(train_loader, total=len(train_loader))
        model.train()
        
        optimizer.zero_grad()
        train_loss_list = []
        for input_ids, attention_mask, p_mask, segment_ids, history_ids, start_position, end_position in train_pbar:
            start_logits, end_logits = model(input_ids=input_ids.to(device), 
                                             segment_ids=segment_ids.to(device), 
                                             attention_mask=attention_mask.to(device))
            total_loss = ###
            total_loss.backward()
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()
            
            train_loss_list.append(float(total_loss))
            train_pbar.set_postfix(start=float(start_loss), end=float(end_loss), total_loss=float(total_loss))
        print("Train loss:", sum(train_loss_list)/len(train_loss_list))
        
        model.eval()
        val_pbar = progress_bar(valid_loader, total=len(valid_loader))

        with torch.no_grad():
            total_loss_list = []
            for input_ids, attention_mask, p_mask, segment_ids, history_ids, start_position, end_position in val_pbar:
                start_logits, end_logits = model(input_ids=input_ids.to(device), 
                                             segment_ids=segment_ids.to(device), 
                                             attention_mask=attention_mask.to(device))

                total_loss = ###
                val_pbar.set_postfix(total_loss=float(total_loss))

                total_loss_list.append(float(total_loss))
        print("Validation loss:", sum(total_loss_list)/len(total_loss_list))
        if dev_loss < sum(total_loss_list)/len(total_loss_list):
            print("Early Stop")
            break
        dev_loss = sum(total_loss_list)/len(total_loss_list)
        
        print(f"Save the model to {model_file}")
        torch.save(model.state_dict(), model_file)
    
print(f"Loading tokenizer {model_name}...")
tokenizer = BertTokenizer.from_pretrained(model_name)
special_tokens_dict = {'additional_special_tokens':['<Q>', '<A>']}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.save_pretrained(tokenizer_dir)
    
train_dataset = Dataset(data_file=train_data,
                        feature_file=train_feature,
                        tokenizer=tokenizer, 
                        mode="train")

valid_dataset = Dataset(data_file=valid_data,
                        feature_file=valid_feature,
                        tokenizer=tokenizer, 
                        mode="valid")

device = torch.device("cuda")
model = CQA(model_name, tokenizer)

fit(model, train_dataset, valid_dataset, device, 
    epochs=args.epochs, batch_size=args.batch_size, warmup_prop=args.warmup_prop, lr=args.learning_rate)

print("Done")