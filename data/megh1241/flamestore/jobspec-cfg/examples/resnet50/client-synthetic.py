import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = ''
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import random
from tensorflow.keras import backend as K
import flamestore
from flamestore.client import Client
import synthetic 
import spdlog
import time
import math
from datetime import datetime
logger = spdlog.ConsoleLogger("Benchmark")
logger.set_pattern("[%Y-%m-%d %H:%M:%S.%F] [%n] [%^%l%$] %v")
tf.debugging.set_log_device_placement(True)

import random
import string
import numpy as np


print(''.join(random.choices(string.ascii_lowercase, k=5)))

store_timings = []
load_timings = []

def gen_random_string():
    return ''.join(random.choices(string.ascii_lowercase, k=5))

def __load_dataset():
    return synthetic.load_dataset(
        train_file='train_signs.h5',
        test_file='test_signs.h5',
        train_set_x='train_set_x',
        train_set_y='train_set_y',
        test_set_x='test_set_x',
        test_set_y='test_set_y',
        list_classes='list_classes')


def create_and_train_new_model(workspace, dataset, size_per_layer, num_layers, variance, num_transferred, file_format='hdf5', file_system='pfs'):
    logger.info('===> Creating FlameStore client')
    client = Client(workspace=workspace)
    logger.info('===> Creating Keras model')
    model = synthetic.create_model(size_per_layer, num_layers, variance, num_transferred)
    logger.info('===> Building model')
    synthetic.build_model(model)
    logger.info('===> Registering model')
    name = gen_random_string()
    ms1 = time.time_ns() / 1000.0
    client.register_model(name, model, include_optimizer=False)
    ms2 = time.time_ns() / 1000.0
    print('register ts: ', end='', flush=True)
    print(ms2 - ms1, flush=True)
    logger.info('===> Saving model data')
    ms1 = time.time_ns() / 1000.0
    if file_system == 'pfs':
        filename = 'my_model'
    else:
        filename = '/dev/shm/my_model'

    if file_format == 'hdf5':
        model.save(filename+'.h5')
    elif file_format == 'savemodel':
        model.save(filename)
    else:
        client.save_weights(name, model, include_optimizer=False)
    ms2 = time.time_ns() / 1000.0
    store_timings.append(ms2-ms1)
    del model
    K.clear_session()
    return name

def reload_and_eval_existing_model(workspace, dataset, name, file_format='hdf5', file_system='pfs'):
    logger.info('===> Creating FlameStore client')
    client = Client(workspace=workspace)
    logger.info('===> Reloading model config')
    model = client.reload_model(name, include_optimizer=False)
    logger.info('===> Rebuilding model')
    synthetic.build_model(model)
    logger.info('===> Reloading model data')
    ms1 = time.time_ns() / 1000.0
    if file_system == 'pfs':
        filename = 'my_model'
    else:
        filename = '/dev/shm/my_model'

    if file_format == 'hdf5':
         tf.keras.models.load_model(filename + '.h5')
    elif file_format == 'savemodel':
         tf.keras.models.load_model(filename)
    else:
        client.load_weights(name, model, include_optimizer=False)
    ms2 = time.time_ns() / 1000.0
    load_timings.append(ms2-ms1)
    del model
    K.clear_session()


def from_human(s: str) -> int:
    if s.endswith("g"):
        return int(s.split("g")[0]) * 1024**3
    elif s.endswith("m"):
        return int(s.split("m")[0]) * 1024**2
    elif s.endswith("k"):
        return int(s.split("k")[0]) * 1024
    else:
        raise RuntimeError(f"unable to parse \"{s}\"")


def duplicate_and_eval_existing_model(workspace, dataset):
    logger.info('===> Creating FlameStore client')
    client = Client(workspace=workspace)
    logger.info('===> Duplicating model')
    client.duplicate_model('my_model', 'my_duplicated_model')
    logger.info('===> Reloading duplicate')
    model = client.reload_model('my_duplicated_model', include_optimizer=False)
    logger.info('===> Rebuilding model')
    synthetic.build_model(model)
    logger.info('===> Reloading model data')
    client.load_weights('my_duplicated_model', model, include_optimizer=False)
    del model
    K.clear_session()


if __name__ == '__main__':
    print("PRINTING GPUs!!!", flush=True)
    print(tf.config.list_physical_devices('GPU'), flush=True)
    random.seed(1234)
    if(len(sys.argv) < 4):
        logger.info("Usage: python client-lenet5.py <workspace> <hdf5/savemodel/flamestore> <pfs/tmpfs>")
        sys.exit(-1)
    dataset = []
    workspace = sys.argv[1]
    file_format = sys.argv[2]
    file_system = sys.argv[3]
    num_layers = int(sys.argv[4])
    variance = int(sys.argv[5])  
    total_size_str = sys.argv[6]
    total_size = from_human(total_size_str)
    relative_size = sys.argv[7]
    size_per_layer = int( (-32 - num_layers + math.sqrt(32**2 + num_layers*total_size)) / (2*num_layers))
    logger.info('=> Workspace is '+workspace)
    logger.info('=> Creating and training a new model')
    name = create_and_train_new_model(workspace, dataset, size_per_layer, num_layers, variance, file_format, file_system)
    logger.info('=> Reloading and evaluating existing model')
    reload_and_eval_existing_model(workspace, dataset, name, file_format, file_system)
    logger.info('=> Duplicating and evaluating a model')
    print('store timings: ', end='', flush=True)
    print(store_timings)
    print('read timings: ', end='', flush=True)
    print(load_timings, flush=True)
    store_filename = dirname + '_store_'+relative_size+'_'+100+'.npy'
    load_filename = dirname + '_store_'+relative_size+'_'+100+'.npy'
    np.savetxt(store_filename, [store_timings, store_timings], delimiter=',', fmt='%d')
    np.savetxt(load_filename, load_timings, delimiter=',', fmt='%d')
