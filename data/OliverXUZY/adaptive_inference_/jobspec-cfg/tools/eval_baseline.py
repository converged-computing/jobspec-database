import random
import argparse
import torch
import numpy as np
from collections import defaultdict
import torch.nn as nn 
from itertools import combinations
from math import comb
from torch.utils.data import Subset

import sys
import os
run_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(run_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from libs.datasets import make_dataset, make_data_loader
from libs.model import make_resnet, make_vit, Evaluator
from libs.utils import fix_random_seed, check_file
from libs.core import load_config
import libs.utils as utils
from libs.utils import Timer, time_str

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def parse_args():
    parser = argparse.ArgumentParser(description="This is my training script.")
    parser.add_argument('-c', '--config', type=str, help='config file path')
    parser.add_argument('-n', '--name', type=str, help='job name')
    parser.add_argument('-g', '--gpu', type=str, default=None, help='GPU IDs')
    parser.add_argument('-m', '--model', type=str, default='resnet50', help='backbone')
    parser.add_argument('--dataset', type=str, default='imagenet', help='The dataset we used')
    parser.add_argument('--skip_block', type=int, default=0, help='how many blocks to skip')
    parser.add_argument('--log_path', type=str, default=None, help='the log path')
    parser.add_argument('--limit', type=int, default=0, help='limit sample to evaluate')


    args = parser.parse_args()
    return args

def main(args):
    branches_per_setting = 256

    if args.model == "resnet18":
        num_block = 8-1
    elif args.model == "resnet50":
        num_block = 16-1
    elif "vit" in args.model:
        num_block = 12-1
    else:
        raise NotImplementedError("num block of other backbone hasn't been implemented yet")

    # skip block
    skip_block = args.skip_block
    print(f"skip {skip_block} block | ")
    num_combinations = comb(num_block, skip_block)

    #######
    if num_combinations <= branches_per_setting:
        all_combinations = list(combinations(range(num_block), skip_block))
        masks = np.ones((len(all_combinations), num_block))
        for i, idx in enumerate(all_combinations):
            masks[i, idx] = 0
    else:

        masks = np.ones((branches_per_setting, num_block))
        # Set random seed for the built-in random module
        seed_value = 42  # you can choose any number you like
        random.seed(seed_value)
        
        for i in range(branches_per_setting): # the last one is always true, skip no blocks
            idx = random.sample(range(num_block), skip_block)
            masks[i, idx] = 0
    #######

    evaluator = Evaluator(model_name = args.model, dataset_name = args.dataset, limit = args.limit, random_seed = 2023)
    if args.log_path is None:
        log_path = f"log/subset{args.limit}/testEval_{args.model}_{args.dataset}"
    else:
        log_path = args.log_path
    evaluator.set_log_path(log_path = log_path)
    evaluator.evaluate(masks, skip_block = skip_block)
    evaluator.save(masks = masks, skip_block = skip_block)

if __name__ == '__main__':
    args = parse_args()
    device_count = torch.cuda.device_count()
    args._parallel = False
    # if device_count > 1:
    #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
    #     args._parallel = True
    main(args)
