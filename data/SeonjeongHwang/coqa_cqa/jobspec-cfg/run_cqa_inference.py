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
from tool.inference_utils import write_predictions

MIN_FLOAT = -1e30

import argparse

parser = argparse.ArgumentParser(description="CQA")

### Arguments for Traning
parser.add_argument("--batch-size", type=int)

### Directories
parser.add_argument("--output-dir", type=str)
parser.add_argument("--result-dir", type=str)

### Arguments for Dataset
parser.add_argument("--num-turn", type=int, default=3)
parser.add_argument("--max-seq-length", type=int, default=512)
parser.add_argument("--max-history-length", type=int, default=128)
parser.add_argument("--doc-stride", type=int, default=192)

parser.add_argument("--model-name", type=str, default="bert-cased-large")

### Inference Setting
parser.add_argument("--n-best-size", type=int, default=5)
parser.add_argument("--max-answer-length", type=int, default=30)

args = parser.parse_args()

    

exp_dir = os.path.join(args.output_dir, args.result_dir)

model_file=exp_dir+"/model/model.pth"
tokenizer_dir=exp_dir+"/tokenizer"

config = exp_dir+"/config.json"
with open(config, "r") as f:
    config_items = json.load(f)

model_name = config_items["model_name"]
max_seq_length = config_items["max_seq_length"]
max_history_length = config_items["max_history_length"]
doc_stride = config_items["doc_stride"]
num_turn = config_items["num_turn"]

test_data = f"data/coqa/coqa-dev-v1.0.json"
test_example = f"data/coqa/dev_{args.num_turn}_examples.pkl"
test_feature = f"data/coqa/dev_{args.num_turn}_features.pkl"
    
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
    def __init__(self, data_file, example_file, feature_file, tokenizer, mode):
        if os.path.exists(example_file):
            print(f"Loading {mode} examples from {example_file}...")
            with open(example_file, "rb") as f:
                self.examples = pickle.load(f)
        else:
            print(f"Generating {mode} examples...")
            self.examples = read_manmade_example(input_file=data_file, is_training=False, num_turn=num_turn)
            print(f"Save the examples to {example_file}...")
            with open(example_file, "wb") as f:
                pickle.dump(self.examples, f, pickle.HIGHEST_PROTOCOL)
                
        if os.path.exists(feature_file):
            print(f"Loading {mode} features from {feature_file}...")
            with open(feature_file, "rb") as f:
                self.features = pickle.load(f)
        else:
            with open(example_file, "wb") as f:
                pickle.dump(self.examples, f, pickle.HIGHEST_PROTOCOL)    
            print(f"Generating {mode} features...")
            self.features = convert_examples_to_features(examples=self.examples, 
                                                         tokenizer=tokenizer, 
                                                         max_seq_length=max_seq_length,
                                                         max_history_length=max_history_length,
                                                         doc_stride=doc_stride,
                                                         is_training=False)
            print(f"Save the features to {feature_file}...")
            with open(feature_file, "wb") as f:
                pickle.dump(self.features, f, pickle.HIGHEST_PROTOCOL)
        
        self.unique_id = self.features["unique_id"]
        self.input_ids = self.features["input_ids"]
        self.attention_mask = self.features["attention_mask"]
        self.segment_ids = self.features["segment_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        unique_id = self.unique_id[idx]
        input_ids = torch.tensor(self.input_ids[idx])
        attention_mask = torch.tensor(self.attention_mask[idx])
        segment_ids = torch.tensor(self.segment_ids[idx])
        
        return input_ids, attention_mask, segment_ids, unique_id

class CQA(nn.Module):
    def __init__(self, bert_model_name, tokenizer):
        super().__init__()
        self.BertEncoder = BertModel.from_pretrained(bert_model_name)
        self.BertEncoder.resize_token_embeddings(len(tokenizer))
        
        ### CODE ###
        
    def forward(self, input_ids, segment_ids, attention_mask, history_ids, p_mask):
        bert_output = self.BertEncoder(input_ids=input_ids, 
                                       attention_mask=attention_mask, 
                                       token_type_ids=segment_ids).last_hidden_state
        
        ### CODE ###
    
def prediction(model, test_dataset, device):
    progress_bar = tqdm.tqdm
    model = model.to(device)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_pbar = progress_bar(test_loader, total=len(test_loader))
    
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits"])
    all_results = []
    print("Predicting answers...")
    for input_ids, attention_mask, p_mask, segment_ids, history_ids, unique_id in test_pbar:
        start_logits, end_logits = model(input_ids=input_ids.to(device), 
                                         segment_ids=segment_ids.to(device), 
                                         attention_mask=attention_mask.to(device))
        
        batch_num = start_logits.size(0)
        for idx in range(batch_num):
            start_logit = [float(x) for x in start_logits[idx].tolist()]
            end_logit = [float(x) for x in end_logits[idx].tolist()]
            
            all_results.append(RawResult(unique_id=int(unique_id[idx]),
                                         start_logits=start_logit,
                                         end_logits=end_logit))
    return all_results
    
print(f"Loading tokenizer from {tokenizer_dir}...")
tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
print(f"Loading trained model from {model_file}...")

device = torch.device("cuda")

model = CQA(model_name, tokenizer, args.batch_size, device)
model.load_state_dict(torch.load(model_file))

test_dataset = Dataset(data_file=test_data,
                       example_file=test_example,
                       feature_file=test_feature,
                       tokenizer=tokenizer, 
                       mode="test")

all_results = prediction(model, test_dataset, device)

output_prediction_file = os.path.join(exp_dir, "predictions.json")
output_nbest_file = os.path.join(exp_dir, "nbest_predictions.json")

print("Writing predictions...")
write_predictions(all_examples=test_dataset.examples,
                  features_dict=test_dataset.features,
                  all_results=all_results,
                  n_best_size=args.n_best_size,
                  max_answer_length=args.max_answer_length,
                  do_lower_case=True,
                  tokenizer=tokenizer,
                  output_prediction_file=output_prediction_file,
                  output_nbest_file=output_nbest_file)

print("Done")