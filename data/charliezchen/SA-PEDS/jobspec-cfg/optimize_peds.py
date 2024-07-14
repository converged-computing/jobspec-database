import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from matplotlib.animation import FuncAnimation, writers

from torch.optim import SGD

from tqdm import tqdm
import json
import pickle
from datetime import datetime

from utils import *

from algo import *
import argparse

import os
import yaml
import pprint
import copy

def parse_args():
    parser = argparse.ArgumentParser(description='Optimize using PEDS method')

    # parser.add_argument("--yaml_config_path", type=str, required=True)

    optim = parser.add_argument_group("optimizer_config")
    model = parser.add_argument_group("model_config")
    exp = parser.add_argument_group("experiment_config")

    optim.add_argument("--class", type=str, required=True)
    optim.add_argument("--lr", type=float, default=0.01)

    model.add_argument("--m", type=int, required=True)
    model.add_argument("--N", type=int, required=True)
    model.add_argument("--alpha", type=float, required=True)
    model.add_argument("--alpha-inc", type=float, required=True)
    model.add_argument("--rv", action='store_true')
    model.add_argument("--upper", type=int, default=10)
    model.add_argument("--lower", type=int, default=-10)
    model.add_argument("--init_noise", type=int, default=10)
    model.add_argument("--shift", type=int, default=2)
    model.add_argument("--independent", action='store_true')

    exp.add_argument("--debug", action='store_true')
    exp.add_argument("--test_function", type=str, required=True)
    exp.add_argument("--sample_size", type=int, default=1000)
    exp.add_argument("--seed", type=int, default=2023)

    args = parser.parse_args()

    groups = [optim, model, exp]
    group_dict = []
    for g in groups:
        dest_list = set([action.dest for action in g._group_actions])
        title = g.title
        group_dict.append([dest_list, title])
    arg_dict = {}
    for k, v in vars(args).items():
        found = False
        for dest_list, title in group_dict:
            if k in dest_list:
                if title not in arg_dict:
                    arg_dict[title] = {}
                arg_dict[title][k] = v
                found = True
                break
        if not found: arg_dict[k] = v


    return arg_dict

arg_dict = parse_args()

# with open(arg_dict['yaml_config_path'], "r") as infile:
#     yaml_config = yaml.full_load(infile)

def pretty_print_dict(data):
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(data)


def run(arg_dict):
    result = copy.deepcopy(arg_dict)
    optimizer_config = arg_dict['optimizer_config']
    model_config = arg_dict['model_config']
    experiment_config = arg_dict['experiment_config']

    test_function_class = eval(experiment_config.pop('test_function'))
    model_class = partial(test_function_class, **model_config)
    base_optim_class = eval(optimizer_config.pop('class'))
    optimizer_class = partial(base_optim_class, **optimizer_config)
    
    result['result'] = experiment(model_class, optimizer_class, 
                        **experiment_config)

    if experiment_config['debug']:
        with open("debug_run.pkl", "wb") as f:
            pickle.dump(result, f)
    
    result['result'] = {k:v for k,v in result['result'].items() \
                        if k in ['success_rate', 'mean_loss', \
                                'mean_iter', 'mean_last_x', 'mean_time']}

    pretty_print_dict(result)


def run_exp(config):
    list_key = []
    list_length = []
    base_config = {}

    for k, v in config.items():
        if isinstance(v, list):
            list_key.append(k)
            list_length.append(len(v))
        else:
            base_config[k] = v
    
    cum_length = [1]
    for length in list_length:
        cum_length.append(cum_length[-1] * length)

    if not list_length:
        run(**base_config)
        exit(0)

    for i in range(np.prod(list_length)):
        con = base_config.copy()

        for j in range(len(list_key)):
            index = i // cum_length[j] % list_length[j]
            key = list_key[j]
            con[key] = config[key][index]
        run(**con)


# for k, v in arg_dict.items():
#     if v is not None:
#         if k in ['N', 'm', 'alpha', 'alpha_inc', 'rv']:
#             yaml_config['model_config'][k] = v
#         elif k in ['test_function', 'debug']:
#             yaml_config['experiment_config'][k] = v
#         elif k in ['yaml_config_path']:
#             pass
#         else:
#             raise TypeError(f"Don't know the key {k}")

# pretty_print_dict(arg_dict)

run(arg_dict)
# run_exp(yaml_config)
# print(yaml_config)


    

    



