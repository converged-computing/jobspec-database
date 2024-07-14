import os
from tensorflow import keras
import tensorflow as tf
import collections
import requests
import json
requests.packages.urllib3.disable_warnings()
import ssl
from tensorflow import keras
import time
import traceback
import argparse
from collections import defaultdict
from typing import List
import random
import psutil
import gc
import uuid
import networkx as nx
import numpy as np
import tensorflow as tf
import pickle
import os, logging
import hashlib
import copy
from datetime import datetime
import random
import ctypes
import argparse
from transfer_methods import * 
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


def _transfer_name():
    return np.uint64(uuid.uuid4().int>>64)


def get_model_size(model):
    siz=0
    dtype_size=0
    for layer in model.layers:
        weights = layer.get_weights()

        for i in weights:
            siz+=i.size
    print("total size: ", flush=True)
    print(siz * 32 / 1000000)


def generate_layer_index_dict():
    #layer_index_dict decides which layer mutate for different % of layers transferred/stored
    layer_index_dict = {'small': {25: 0, 50: 0, 75: 0, 100:0}, 'large': {25: 0, 50: 0, 75: 0}}
    layers_transferred = [25, 50, 75]
    curr_index_small = 2
    curr_index_large = 4
    for lt in layers_transferred:
        layer_index_dict['small'][lt] = curr_index_small
        layer_index_dict['large'][lt] = curr_index_large
        curr_index_small += 1
        curr_index_large += 3
    return layer_index_dict


def _standardize_names(model):
    # first build the graph of the layers
    graph = collections.defaultdict(list)
    hashed_names = collections.defaultdict(str)
    counts = collections.defaultdict(int)
    if isinstance(model, tf.keras.Sequential):
        #functional model doesn't have 'inbound_nodes', simply iterate an build graph linearly
        for last_layer, current_layer in zip(model.layers, model.layers[1:]):
            graph[current_layer.name] = [last_layer.name]
    elif isinstance(model, tf.keras.Model):

        #assume instance of functional model
        model_config = model.get_config()
        for layer in model_config['layers']:
            if len(layer['inbound_nodes']) > 0:
                graph[layer['name']] = [
                            inbound[0] for inbound in layer['inbound_nodes'][0]
                        ]

    def inner(layer):
        #we can't assume that layers aren't shared, so force a copy
        layer_config = copy.deepcopy(layer.get_config())

        #we need the old name to lookup predecessors
        old_name = layer_config['name']

        #we don't want to hash the name since this would make things recursively defined
        del layer_config['name']

        begin = time.perf_counter()
        layer_hash = hashlib.sha3_512()
        layer_hash.update(json.dumps(layer_config, sort_keys=True).encode())
        for pred_name in graph[old_name]:
            layer_hash.update(hashed_names[pred_name].encode())
        layer_hash = layer_hash.hexdigest()
        end = time.perf_counter()

        base_name = layer.__class__.__name__ + layer_hash
        hashed_names[old_name] = base_name + "_" + str(counts[base_name])
        counts[base_name]+=1
        #create the new layer with the appropriate name
        layer_config['name'] = hashed_names[old_name]
        return layer.__class__.from_config(layer_config)
    return inner


def standardize_names(model: tf.keras.Model):
    return tf.keras.models.clone_model(model, input_tensors=model.inputs, clone_function=_standardize_names(model))


def _mutate_standardize_names(model, layer_to_mutate_index):
    layer_to_mutate_name = model.layers[layer_to_mutate_index].get_config()['name']
    def inner(layer):
        layer_config = copy.deepcopy(layer.get_config())
        if layer_config['name'] == layer_to_mutate_name:
            old_name = layer_config["name"]
            del layer_config["name"]
            layer_config["name"] =  str(uuid.uuid4().int>>64)
        return layer.__class__.from_config(layer_config)
    return inner


def mutate_standardize_names(model, layer_to_mutate_index):
    return tf.keras.models.clone_model(model, input_tensors=model.inputs, clone_function=_mutate_standardize_names(model, layer_to_mutate_index))


def create_keras_model(size='small', use_bias=True):
    num_units=0
    if size=='small':
        num_layers = 4
        dense_layer_activation="relu"
        num_units=1007
    else:
        num_units=1631
        num_layers = 12
        dense_layer_activation="tanh"

    inputs = keras.Input((num_units,))
    x = keras.layers.Dense(num_units, activation=dense_layer_activation, use_bias=use_bias, kernel_initializer='random_normal', bias_initializer='zeros')(inputs)
    for i in range(num_layers-1):
        x = keras.layers.Dense(num_units, activation=dense_layer_activation, use_bias=use_bias, kernel_initializer='random_normal',
            bias_initializer='zeros')(x)
    outputs = keras.layers.Dense(2, activation=dense_layer_activation, use_bias=use_bias, kernel_initializer='random_normal',
            bias_initializer='zeros')(x)
    model = keras.Model(inputs, outputs)
    #get_model_size(model)
    return model


def create_model(size='small', use_bias=True, percent_transfer=25):
    model = create_keras_model(size=size, use_bias=use_bias) 
    model = standardize_names(model)
    model_id = _transfer_name()
   
    if percent_transfer == 0 or percent_transfer==100:
        return model_id, model
   
    layer_index = generate_layer_index_dict()
    model = mutate_standardize_names(model, layer_index[size][percent_transfer]) 
    
    return model_id, model


def zero_center(arr):
    first = arr[0]
    zero_center_arr = [ele - first for ele in arr]
    return zero_center_arr


def append_results_to_dict(ts, results_dict, perc_transfered, action='transfer', size='small'):
    ts_zero_centered = zero_center(ts)
    search_key = action + '_' + size + '_'+ str(perc_transfered)
    if search_key not in results_dict:
        results_dict[search_key] = []
    results_dict[search_key].append(ts_zero_centered)


def save_to_file(results_dict, save_dir):
    for i in results_dict:
        mean = np.mean(results_dict[i], axis=0)
        std = np.std(results_dict[i], axis=0)
        to_save = np.array([mean, std])
        action=''
        if 'store' in i:
            action = 'store'
        else:
            action = 'transfer'

        with open(save_dir + i+'.npy', 'wb') as f:
            np.save(f, to_save)


transfer_method = DataStatesModelRepo()
parser = argparse.ArgumentParser()
parser.add_argument("--save_directory", type=str, default='.')
parser.add_argument("--number_of_ops_per_size", type=int, default=30)
parser.add_argument("--rand_seed", type=int, default=10)

args = parser.parse_args()
save_dir = args.save_directory
num_trials = args.number_of_ops_per_size
size_arr = ['small', 'large']
perc_transferred_arr = [0, 25, 50, 75, 100]

perc_transferred_random_trials = random.choices(perc_transferred_arr, k=num_trials )

results_dict = {}
for size in size_arr:
    model_id, model = create_model(size=size, percent_transfer = 0)
    suffix = [layer.name for layer in model.layers]
    transfer_method.store(model_id, model, suffix)
    ts = transfer_method.get_time_stamps()
    transfer_method.clear_time_stamps()
    cid = ctypes.c_uint64(model_id)
    del model
    tf.keras.backend.clear_session()
    for percent_transfer in perc_transferred_random_trials:
        model_id, model = create_model(size=size, percent_transfer=percent_transfer)
        if percent_transfer > 0:
            transferred, transferred_id = transfer_method.transfer(model_id, model)
            ts = transfer_method.get_time_stamps()
            append_results_to_dict(ts, results_dict, percent_transfer, action='transfer', size=size)
            transfer_method.clear_time_stamps()
        else:
            transferred = []

        suffix = [layer.name for layer in model.layers if layer.name not in transferred] 
        if suffix:
            transfer_method.store(model_id, model, suffix)
            ts = transfer_method.get_time_stamps()
            transfer_method.clear_time_stamps()
            append_results_to_dict(ts, results_dict, 100-percent_transfer, action='store', size=size)
        del model
        tf.keras.backend.clear_session()


save_to_file(results_dict, save_dir)
