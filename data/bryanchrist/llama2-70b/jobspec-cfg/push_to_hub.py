import os
import sys
import builtins
#os.environ['TRANSFORMERS_CACHE'] = '/project/SDS/research/christ_research/Llama 2/llama2-70b/cache'
os.environ['TRANSFORMERS_CACHE'] = '/scratch/brc4cb/llama/cache'
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import random

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from datasets import load_dataset, Dataset
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
#from adapter-transformers import AdapterType, AdapterConfig, load_adapter

# Set the environment variable
os.environ["HF_REMOTES_OFFLINE"] = "1"
from dotenv import load_dotenv
# Load the environmental variables from the .env file
load_dotenv()

token = os.getenv('huggingface_token')
if token:
    print('token loaded')
    
token = os.environ['huggingface_token']
#Uncomment if needed
from huggingface_hub import login
login(token=token)

# Redirect stdin to /dev/null
sys.stdin = open(os.devnull)

model_path = "meta-llama/Llama-2-70b-hf"   # Specify the path to the model
adapter_path = "mathwell/llama_QA_adapter_no_embed/checkpoint-1250"  # Specify the path to the adapter weights
tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True)

# Patch the built-in input function to return 'y' automatically
def mock_input(prompt=None):
    return 'y'

# Patch the input function to use the mock_input function
builtins.input = mock_input

try:
    # Attempt to load the model with trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True, 
       # max_memory=max_memory,
        torch_dtype=torch.float16,
        use_auth_token=True,
        config=AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    )
except EOFError:
    # If an EOFError occurs, provide the expected input ('y')
    pass

# Restore stdin
sys.stdin = sys.__stdin__

# Load the adapter weights
model = PeftModel.from_pretrained(model, adapter_path)
url = "https://huggingface.co/bryanchrist/MATHWELL"
model.push_to_hub(repo_url=url)
tokenizer.push_to_hub(repo_url=url)
