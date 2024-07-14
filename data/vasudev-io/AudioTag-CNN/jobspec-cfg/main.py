import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataload import MagnaTagATune, SpecMagnaTagATune
import argparse
from pathlib import Path
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from misc import get_summary_writer_log_dir, compute_global_min_max, save_min_max_values, load_min_max_values, get_val_key, get_test_key
from models import BaseCNN, ChunkResCNN, CRNN
from train import Trainer

torch.backends.cudnn.benchmark = True

default_dataset_dir = str(Path.home() / ".cache" / "torch" / "datasets")

#change print flags for relevant arguments
arguments = [
    {"name": "--dataset-root", "default": default_dataset_dir, "print_flag": 0},
    {"name": "--log-dir", "default": str(Path("logs")), "type": Path, "print_flag": 0},
    {"name": "--sgd-momentum", "default": 0.93, "type": float, "print_flag": 1},
    {"name": "--learning-rate", "default": 0.05, "type": float, "print_flag": 1},
    {"name": "--conv-length", "default": 256, "type": int, "print_flag": 1},
    {"name": "--conv-stride", "default": 256, "type": int, "print_flag": 1},
    {"name": "--batch-size", "default": 10, "type": int, "print_flag": 1},
    {"name": "--val-frequency", "default": 1, "type": int, "print_flag": 0},
    {"name": "--log-frequency", "default": 100, "type": int, "print_flag": 0},
    {"name": "--print-frequency", "default": 100, "type": int, "print_flag": 0},
    {"name": "--worker-count", "default": 1, "type": int, "print_flag": 0},
    {"name": "--epochs", "default": 30, "type": int, "print_flag": 1},
    {"name": "--weight-decay", "default": 1e-5, "type": float, "print_flag": 1},
    {"name": "--model", "default": "ChunkResCNN", "type": str, "print_flag": 1},
    {"name": "--op", "default": "sgd", "type": str, "print_flag": 1},
    {"name": "--dropout", "default": 0.5, "type": float, "print_flag": 1},
    {"name": "--data-mode", "default": 1, "type": float, "print_flag": 1},
    {"name": "--scheduler", "default": 1, "type": float, "print_flag": 1},
]

parser = argparse.ArgumentParser(
    description="Train a CNN on MagnaTagATune.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

print_flags = {}

for arg in arguments:
    parser.add_argument(arg["name"], default=arg["default"], type=arg.get("type", str))
    print_flags[arg["name"]] = arg["print_flag"]

args = parser.parse_args()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

#load data
def load_data(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)  
data_path = 'datasets.json'
dataset = load_data(data_path)
global_min, global_max = load_min_max_values('min_max_values.json')

def main(args):

    args.dataset_root = Path(args.dataset_root)
    args.dataset_root.mkdir(parents=True, exist_ok=True)

    #uncomment below if the min_max_values.json file doesn't exist
    '''
    train_dataset= MagnaTagATune(dataset_path='/mnt/storage/scratch/ee20947/MagnaTagATune/annotations/train.pkl', samples_path='/mnt/storage/scratch/ee20947/MagnaTagATune/samples')
    if global_min is None or global_max is None:
        # File doesn't exist, compute and save the values
        #global_min, global_max = compute_global_min_max(train_dataset)
        save_min_max_values(global_min, global_max, 'min_max_values.json')
    '''

    if args.model == "BaseCNN":
        
        train_dataset = MagnaTagATune('trainLocal', dataset, global_min, global_max)
        val_dataset = MagnaTagATune('valLocal', dataset, global_min, global_max)
        test_dataset = MagnaTagATune('testLocal', dataset, global_min, global_max)
        model = BaseCNN(args.conv_length, args.conv_stride)
                        
    if args.model == "ChunkResCNN1":

        train_dataset = SpecMagnaTagATune('trainSpec', dataset)
        val_dataset = SpecMagnaTagATune('valSpec', dataset)
        test_dataset = SpecMagnaTagATune('testSpec', dataset)
        model = ChunkResCNN()

    if args.model == "ChunkResCNN1" and args.data_mode == 2:

        train_dataset = SpecMagnaTagATune('trainSpec1', dataset)
        val_dataset = SpecMagnaTagATune('valSpec1', dataset)
        test_dataset = SpecMagnaTagATune('testSpec1', dataset)
        model = ChunkResCNN()

    if args.model == "ChunkResCNN1" and args.data_mode == 3:

        train_dataset = SpecMagnaTagATune('trainSpec2', dataset)
        val_dataset = SpecMagnaTagATune('valSpec2', dataset)
        test_dataset = SpecMagnaTagATune('testSpec2', dataset)
        model = ChunkResCNN()

    if args.model == "ChunkResCNN1" and args.data_mode == 4:

        train_dataset = SpecMagnaTagATune('trainSpec3', dataset)
        val_dataset = SpecMagnaTagATune('valSpec3', dataset)
        test_dataset = SpecMagnaTagATune('testSpec3', dataset)
        model = ChunkResCNN()

    if args.model == "ChunkResCNN2" and args.data_mode == 5:

        train_dataset = SpecMagnaTagATune('trainSpecChunk', dataset)
        val_dataset = SpecMagnaTagATune('valSpecChunk', dataset)
        test_dataset = SpecMagnaTagATune('testSpecChunk', dataset)
        model = ChunkResCNN()

    if args.model == "ChunkResCNN2" and args.data_mode == 6:

        train_dataset = SpecMagnaTagATune('trainSpecChunk1', dataset)
        val_dataset = SpecMagnaTagATune('valSpecChunk1', dataset)
        test_dataset = SpecMagnaTagATune('testSpecChunk1', dataset)
        model = ChunkResCNN()
        
    if args.model == "CRNN":
            
        train_dataset = SpecMagnaTagATune('trainSpec', dataset)
        val_dataset = SpecMagnaTagATune('valSpec', dataset)
        test_dataset = SpecMagnaTagATune('testSpec', dataset)
        model = CRNN()

    #intialise everything for training
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker_count)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker_count)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.worker_count)
    criterion = nn.BCELoss()
    
    if args.op == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.sgd_momentum, dampening=0, weight_decay=args.weight_decay , nesterov=False)
    elif args.op == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, betas=(0.95, 0.999))

    if args.scheduler == 1:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.0, patience=2, verbose=False)

    log_dir = get_summary_writer_log_dir(args, arguments)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(str(log_dir), flush_secs=5)
    trainer = Trainer(
        model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, summary_writer, DEVICE
    )

    keyVal = get_val_key(args.model)

    #training (validation occurs within)
    trainer.train(args.epochs, args.val_frequency, print_frequency=args.print_frequency, log_frequency=args.log_frequency, key=keyVal)

    keyTest = get_test_key(args.model)

    #test
    trainer.validate(test_loader, dataset, keyTest , "Test: ")

    summary_writer.close()

if __name__ == "__main__":
    main(parser.parse_args()) 

