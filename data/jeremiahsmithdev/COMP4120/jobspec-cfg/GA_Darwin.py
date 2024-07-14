from keras import backend as K
#K.tensorflow_backend._get_available_gpus()
import tensorflow as tf
print(tf.test.gpu_device_name())
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

# RMSProp(learning rate)
# Temperature
# Pickle
# manual deap


import tensorflow as tf
# Importing required packages
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as split

from keras.utils import multi_gpu_model

from keras.layers import CuDNNLSTM, Input, Dense, Dropout
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray

import multiprocessing
import pickle                   # evolutionary checkpointing

import sys
import collections

def ic(text):
    # remove all non alpha and whitespace and force uppercase
    flattext = "".join([x.upper() for x in text.split() if x.isalpha()])
    N = len(flattext)
    freqs = collections.Counter(flattext)
    alphabet = map(chr, range(ord('A'), ord('Z')+1))
    freqsum = 0.0

    # math
    for letter in alphabet:
        freqsum += freqs[letter] * (freqs[letter] - 1)
    if N == 0:
        return 1.067
    if N == 1:
        return 1.0
    IC = freqsum / (N*(N-1))

    return IC

np.random.seed(1120)

import keras as keras
#print("NGPUs: ", str(len(keras.backend._get_available_gpus())))

# QUICK ACCESS training params
ngpu = 1    # per process?
mp = 0      # multiprocessing on/off
nprocesses = 1


# 1. READING DATASET
    # load ascii text and convert to lowercase
#filename = "OriginOfSpecies.txt"
#filename = "OnTheOriginOfSpecies.txt"
filename = "sample.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

    # create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

    #summarize the loaded sata
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)


# prepare the dataset of input to output pairs encoded as integers

# DEFINING HELPER FUNCTIONS
def prepare_dataset(seq_length):
    dataX = []
    dataY = []
    # prepare the dataset of input to output pairs encoded as integers
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)
    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    Y = np_utils.to_categorical(dataY)
    return X, Y, dataX, dataY

def train_evaluate(ga_individual_solution):
    # define the LSTM model
    print (ga_individual_solution)
    # Decode GA solution to integer for window_size and num_units
    #seq_length_bits = BitArray(ga_individual_solution[0:4])         # 8-136
    #RNN_size_bits = BitArray(ga_individual_solution[4:6])          # 128-512
    RNN_size_bits = BitArray(ga_individual_solution[0:9])          # 128-512
    #RNN_layers_bits = BitArray(ga_individual_solution[6:8])       # 1-3
    RNN_layers_bits = BitArray(ga_individual_solution[9:11])       # 1-3
    dropout_bits = BitArray(ga_individual_solution[8:13]);      # 0.1 0.2 0.3 0.4
    learning_rate_bits = BitArray(ga_individual_solution[13:16])    # 0-8
    #decay_rate_bits = BitArray(ga_individual_solution[13:16])       # 0-8

    #seq_length = (seq_length_bits.uint * 8) + 8
    RNN_size = RNN_size_bits.uint#(RNN_size_bits.uint * 128) + 128
    RNN_layers = RNN_layers_bits.uint # additional hidden layers
    RNN_layers = 0
    #dropout = (dropout_bits.uint*0.1) + 0.1
    #learning_rate = 0.005 + learning_rate_bits.uint * 0.005 # 0.005, 0.01, 0.015, 0.02 - 0.045
    seq_length = 100
    dropout = 0.2 + dropout_bits.uint * 0.01
    #dropout = 0.5
    learning_rate = 0.01 + learning_rate_bits.uint * 0.001

    #decay_rate = 1 - decay_rate_bits.uint * 0.01       # 0.92 - 1.0
    print('\nseq_length: ', seq_length, ', RNN Size: ', RNN_size, 'RNN Layers:' , RNN_layers+1, 'Droput rate: ', dropout, 'Learning Rate: ', learning_rate)

    X, Y, dataX, dataY = prepare_dataset(seq_length)
    # X_train, X_val, y_train, y_val = split(X, Y, test_size = 0.20, random_state = 1120)

    # define the LSTM model
    model = Sequential()
    # input layer
    model.add(CuDNNLSTM(RNN_size, input_shape=(X.shape[1], X.shape[2]),
                   return_sequences=True))
    # hidden layers
    model.add(Dropout(dropout))

    for x in range(0, RNN_layers):
        model.add(CuDNNLSTM(RNN_size, return_sequences=True))
        model.add(Dropout(dropout))

    model.add(CuDNNLSTM(RNN_size))
    model.add(Dropout(dropout))

    # output layer
    model.add(Dense(Y.shape[1], activation='softmax'))
    
    opt = optimizers.RMSprop(lr=learning_rate)#, decay=decay_rate)
    filepath='weights/weights-{}'.format(ga_individual_solution)#"weights/weights-improvement-{ga_individual_solution}.hdf5"
    #model.compile(loss='categorical_crossentropy', optimizer=opt)

    # add multi gpu model
    #if ngpu > 1:
    #    model = multi_gpu_model(model, gpus=ngpu)
    
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    #model.compile(loss='categorical_crossentropy', optimizer='adam')
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # fit the model
    #with tf.device('/gpu:4'):
    model.fit(X, Y, epochs=20, batch_size = 128*ngpu, callbacks=callbacks_list)#batch_size=batch_size * NUM_GPU
    #print(model.loss.value)

    # predict (generate text)
        # pick a random seed
    start = np.random.randint(0, len(dataX)-1)
    start = 1000       # start seed from 0 for consistency in text generation
    pattern = dataX[start]

    print ("Seed:")
    print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

    # generate characters
    textgen = ""
    # print ("Generating 1000:")
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        textgen += result
        # sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        # print ("\nDone.")

    print("textgen: " + textgen)

    # evaluate fitness
    IC = ic(textgen)
    print("IC Diff: " + str(IC))
    MAX = max(0.067, IC)
    MIN = min(0.067, IC)

    return MAX - MIN,
    #rmse = np.sqrt(mean_squared_error(dataY, Y))
    #print('Validation RMSE: ', rmse,'\n')
    #return rmse,



# CREATE GENETIC ALGORITHM
population_size = 5
num_generations = 3
gene_length = 16
CXPB = 0.4
MUTPB = 0.1

# use -1.0 to represent minimal fitness scores as better, (1.0 to maximize)
creator.create('FitnessMax', base.Fitness, weights = (-1.0,))
creator.create('Individual', list , fitness = creator.FitnessMax)

toolbox = base.Toolbox()
if mp == 1:
    pool = multiprocessing.Pool(nprocesses)
toolbox.register('binary', bernoulli.rvs, 0.5)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n = gene_length)
toolbox.register('population', tools.initRepeat, list , toolbox.individual)
if mp == 1:
    toolbox.register("map", pool.map)

toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.6)
toolbox.register('select', tools.selRoulette)
print("before train evaluate")
toolbox.register('evaluate', train_evaluate)
print("finished train_evaluate")

population = toolbox.population(n = population_size)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
print("\nBEFORE r = algorithms...")

# RUN GENETIC ALGORITHM

r, logbook = algorithms.eaSimple(population, toolbox, cxpb = 0.4, mutpb = 0.1, ngen = num_generations, stats=stats, halloffame=hof, verbose = True)

print("\nAFTER r = algorithms...")
print("\nLogbook:")
print(logbook)
print("r:")
print(r)

"""""
for gen in range(0, num_generations):
    population = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
            
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    #with tf.device('/gpu:4'):
    #for i in range(len(invalid_ind)):
    #    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind[i])
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(population)
    record = stats.compile(population)
    logbook.record(gen=gen, evals=len(invalid_ind), **record)

    # new population
    population = toolbox.select(population, k=len(population))
    """""
