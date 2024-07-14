import argparse

parser = argparse.ArgumentParser(prog = 'Run Model')
parser.add_argument('--dimensionality', type=int, choices=[2, 3], default=3)
parser.add_argument('--removed_average', type=int, choices=[0, 1], default=1)
parser.add_argument('--Zmax', type=int, default=30)
parser.add_argument('--data_location', type=str, default="data/")
parser.add_argument('--saving_location', type=str, default="models/")
parser.add_argument('--model', type=str, default="RNN.SummarySpace3D")
parser.add_argument('--HyperparameterIndex', type=int, choices=range(576), default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--multi_gpu_correction', type=int, choices=[0, 1, 2], default=0, help="0-none, 1-batch_size, 2-learning_rate")
parser.add_argument('--file_prefix', type=str, default="")

inputs = parser.parse_args()
inputs.removed_average = bool(inputs.removed_average)
inputs.model = inputs.model.split('.')
print("INPUTS:", inputs)

import copy
import itertools
import sys
import importlib
import numpy as np
import math
import tensorflow as tf
# import keras
from tensorflow import keras
# import horovod.tensorflow.keras as hvd

# if inputs.gpus == 1:
# #setting up GPU
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1. #setting the percentage of GPU usage
# config.gpu_options.visible_device_list = "0" #for picking only some devices
# config.gpu_options.allow_growth = True
# # config.log_device_placement=True
# keras.backend.set_session(tf.Session(config=config))
# elif inputs.gpus > 1:
#     #init Horovod
#     hvd.init()
#     # Horovod: pin GPU to be used to process local rank (one GPU per process)
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     config.gpu_options.visible_device_list = str(hvd.local_rank())
#     keras.backend.set_session(tf.Session(config=config))
# else:
#     raise ValueError('number of gpus shoud be > 0')
# keras.backend.set_image_data_format('channels_last')

import src.py21cnn.utilities as utilities
# ModelClassObject = getattr(importlib.import_module(f'src.py21cnn.architectures.{inputs.model[0]}'), inputs.model[1])

def leakyrelu(x):
    return keras.activations.relu(x, alpha=0.1)

HyP = {}
HyP["Loss"] = [[None, "mse"]]
HyP["Epochs"] = [inputs.epochs]
HyP["BatchSize"] = [20]
HyP["LearningRate"] = [0.01, 0.001, 0.0001]
HyP["Dropout"] = [0.2, 0.5]
HyP["ReducingLR"] = [True]
HyP["BatchNormalization"] = [True, False]
HyP["Optimizer"] = [
                    [keras.optimizers.RMSprop, "RMSprop", {}],
                    [keras.optimizers.SGD, "SGD", {}],
                    [keras.optimizers.SGD, "Momentum", {"momentum":0.9, "nesterov":True}],
                    # [keras.optimizers.Adadelta, "Adadelta", {}],
                    # [keras.optimizers.Adagrad, "Adagrad", {}],
                    [keras.optimizers.Adam, "Adam", {}],
                    # [keras.optimizers.Adam, "Adam", {"amsgrad":True}],
                    [keras.optimizers.Adamax, "Adamax", {}],
                    [keras.optimizers.Nadam, "Nadam", {}],
                    ]
HyP["ActivationFunction"] = [
                            ["relu", {"activation": keras.activations.relu, "kernel_initializer": keras.initializers.he_uniform()}],
                            # [keras.layers.LeakyReLU(alpha=0.1), "leakyrelu"],
                            ["leakyrelu", {"activation": leakyrelu, "kernel_initializer": keras.initializers.he_uniform()}],
                            ["elu", {"activation": keras.activations.elu, "kernel_initializer": keras.initializers.he_uniform()}],
                            ["selu", {"activation": keras.activations.selu, "kernel_initializer": keras.initializers.lecun_normal()}],
                            # [keras.activations.exponential, "exponential"],
                            # [keras.activations.tanh, "tanh"],
                            ]

Data = utilities.Data(filepath=inputs.data_location, 
                      dimensionality=inputs.dimensionality, 
                      removed_average=inputs.removed_average, 
                      Zmax=inputs.Zmax)
Data.loadTVT(model_type=inputs.model[0])
print("DATA:", str(Data))
HyP_list = list(itertools.product(*HyP.values()))

model_scores = []
for i, h in enumerate(HyP_list):
    HP_dict = dict(zip(HyP.keys(), h))
    HP = utilities.AuxiliaryHyperparameters(**HP_dict)
    filepath = f"{inputs.saving_location}{inputs.file_prefix}{inputs.model[0]}_{inputs.model[1]}_{HP.hash()}_{Data.hash()}"
    # try:
    #     custom_obj = {}
    #     custom_obj["R2"] = utilities.R2
    #     #if activation is leakyrelu add to custom_obj
    #     if HP.ActivationFunction[0] == "leakyrelu":
    #         custom_obj[HP.ActivationFunction[0]] = HP.ActivationFunction[1]["activation"]
    #     model = keras.models.load_model(f"{filepath}_best.hdf5", custom_objects=custom_obj)
    # except:
    #     continue
    try:
        prediction = np.load(f"{filepath}_prediction.npy")
    except:
        continue
    # prediction = model.predict(Data.X['test'], verbose=False)
    R2_score = utilities.R2_final(Data.Y['test'], prediction)
    R2_scores = [utilities.R2_numpy(Data.Y['test'][:, k], prediction[:, k]) for k in range(4)]
    
    nans = 0
    nans += math.isnan(R2_score)
    nans += sum([math.isnan(k) for k in R2_scores])
    if nans:
        continue

    model_scores.append([R2_score, R2_scores, i, str(HP)])
    # print(i)

model_scores.sort(key = lambda x: x[0], reverse=True)
print("ALL MODELS")
for i in model_scores:
    print(i)
print('\n')
print("ONLY INDEXES")
for i in model_scores:
    print(i[2], end =",")
print('\n')
print("BEST 80 INDEXES")
for i in range(80):
    print(model_scores[i][2], end =",")
print('\n')
good_models_5 = set()
good_models_10 = set()

for i in range(4):
    print('\n')
    print(f"BEST 10 PER DIMENSION {i}")
    model_scores.sort(key = lambda x: x[1][i], reverse=True)
    for j in range(10):
        print(model_scores[j])
        good_models_10.add(model_scores[j][2])
        if j < 5:
            good_models_5.add(model_scores[j][2])
print("BEST 10")
for i in good_models_10:
    print(i, end=",")
print("BEST 5")
for i in good_models_5:
    print(i, end=",")
