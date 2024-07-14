import argparse
parser = argparse.ArgumentParser(prog = 'Run Model')
parser.add_argument('--dimensionality', type=int, choices=[2, 3], default=3)
parser.add_argument('--removed_average', type=int, choices=[0, 1], default=1)
parser.add_argument('--Zmax', type=int, default=30)
parser.add_argument('--data_location', type=str, default="data/")
parser.add_argument('--saving_location', type=str, default="models/")
parser.add_argument('--logs_location', type=str, default="logs/")
parser.add_argument('--model', type=str, default="RNN.SummarySpace3D")
# parser.add_argument('--HyperparameterIndex', type=int, choices=range(576), default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--max_epochs', type=int, default=-1)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--multi_gpu_correction', type=int, choices=[0, 1, 2], default=0, help="0-none, 1-batch_size, 2-learning_rate")
parser.add_argument('--file_prefix', type=str, default="")
parser.add_argument('--patience', type=int, default=10)

inputs = parser.parse_args()
inputs.removed_average = bool(inputs.removed_average)
inputs.model = inputs.model.split('.')
if inputs.max_epochs == -1:
    inputs.max_epochs = inputs.epochs
elif inputs.max_epochs > inputs.epochs:
    raise ValueError("max_epochs shoudln't be larger than epochs")
print("INPUTS:", inputs)

import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
if inputs.gpus == 1:
    # #setting up GPU
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1. #setting the percentage of GPU usage
    config.gpu_options.visible_device_list = "0" #for picking only some devices
    config.gpu_options.allow_growth = True
    # config.log_device_placement=True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    # tf.compat.v1.enable_eager_execution(config=config)
elif inputs.gpus > 1:
    import horovod.tensorflow.keras as hvd
    #init Horovod
    hvd.init()
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    # tf.compat.v1.enable_eager_execution(config=config)
else:
    raise ValueError('number of gpus shoud be > 0')

from tensorflow import keras
keras.backend.set_image_data_format('channels_last')

import copy
import itertools
import sys
import importlib
import numpy as np

import src.py21cnn.utilities as utilities
ModelClassObject = getattr(importlib.import_module(f'src.py21cnn.architectures.{inputs.model[0]}'), inputs.model[1])

def leakyrelu(x):
    return keras.activations.relu(x, alpha=0.1)

HP_dict = {
    "Loss": [None, "mse"],
    "Epochs": inputs.epochs,
    "BatchSize": 20,
    "LearningRate": 1e-4,
    "Dropout": 0.5,
    "ReducingLR": True,
    "BatchNormalization": True,
    "Optimizer": [keras.optimizers.Adam, "Adam", {}],
    "ActivationFunction": ["relu", {"activation": keras.activations.relu, "kernel_initializer": keras.initializers.he_uniform()}]
}

HP = utilities.AuxiliaryHyperparameters(**HP_dict)
#creating HP dict for TensorBoard with only HP that are changing and only human-readable information
HP_TensorBoard = {
    "Model": f"{inputs.model[0]}_{inputs.model[1]}",
    "LearningRate": HP_dict["LearningRate"],
    "Dropout": HP_dict["Dropout"],
    "BatchSize": HP_dict["BatchSize"],
    "BatchNormalization": HP_dict["BatchNormalization"],
    "Optimizer": HP_dict["Optimizer"][1],
    "ActivationFunction": HP_dict["ActivationFunction"][0],
}

Data = utilities.Data(filepath=inputs.data_location, 
                      dimensionality=inputs.dimensionality, 
                      removed_average=inputs.removed_average, 
                      Zmax=inputs.Zmax)
Data.loadTVT(model_type=inputs.model[0])

print("HYPERPARAMETERS:", str(HP))
print("DATA:", str(Data))
if inputs.gpus > 1:
    print("HVD.SIZE", hvd.size())

ModelClass = ModelClassObject(Data.shape, HP)
ModelClass.build()
if inputs.gpus == 1:
    utilities.run_model(model = ModelClass.model, 
                        Data = Data, 
                        AuxHP = HP,
                        HP_TensorBoard = HP_TensorBoard,
                        inputs = inputs)
else:
    #corrections for multigpu
    AuxHP = copy.deepcopy(HP)
    if inputs.multi_gpu_correction == 2:
        AuxHP.Optimizer[2]["lr"] *= hvd.size()
    elif inputs.multi_gpu_correction == 1:
        AuxHP.BatchSize //= hvd.size()
    AuxHP.Epochs //= hvd.size()
    AuxHP.MaxEpochs //=hvd.size()
    print("BEFORE RUN AuxHP: ", str(AuxHP))
    print("BEFORE RUN HP: ", str(HP))

    utilities.run_multigpu_model(model = ModelClass.model, 
                                Data = Data, 
                                AuxHP = AuxHP,
                                HP = HP,
                                HP_TensorBoard = HP_TensorBoard,
                                inputs = inputs,
                                # hvd = hvd,
                                )