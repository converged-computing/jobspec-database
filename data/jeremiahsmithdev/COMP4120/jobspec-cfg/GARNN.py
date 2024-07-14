# Importing required packages
import numpy as np
import pandas as pd
import multiprocessing # multiprocessing

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as split

from keras.layers import LSTM, Input, Dense, Dropout
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from bitstring import BitArray
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices()) # Prints out devices that are being used

np.random.seed(1120)

individual_id = 0 # used to name all of the individual attempts and then they
                    # are logged

# 1. READING DATASET
# load ascii text and convert to lowercase
filename = "Darwin.txt"
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
    # Decode GA solution
    seq_length_bits = BitArray(ga_individual_solution[0:4])         # 8 / ... / 136  4b
    RNN_size_bits = BitArray(ga_individual_solution[4:6])          # 128 / 256 / 384 / 512 2b
    layer_count_bits = BitArray(ga_individual_solution[6:8])       # 1 / 1 / 2 / 3 2b
    dropout_bits = BitArray(ga_individual_solution[8:10])       # 0.2 0.3 0.4 0.5 2b

    seq_length = (seq_length_bits.uint * 8) + 8
    RNN_size = (RNN_size_bits.uint * 128) + 128
    layer_count = layer_count_bits.uint

    dropout = (dropout_bits.uint*0.15) + 0.1

    if layer_count == 0:
        layer_count = 1

    print('\nseq_length: ', seq_length, ', RNN Size: ', RNN_size,
          'layer_count: ', layer_count, 'dropout: ',
          dropout)

    X, Y, dataX, dataY = prepare_dataset(seq_length)

    # define the LSTM model
    model = Sequential()

    # input layer
    model.add(LSTM(RNN_size, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
    model.add(Dropout(dropout))

    for x in range(0, layer_count):
        model.add(LSTM(RNN_size))
        model.add(Dropout(dropout))

    # output layer
    model.add(Dense(Y.shape[1], activation='softmax'))

    global individual_id

    # optimize model
    #opt = optimizers.SGD(lr=learning_rate, decay=decay_rate)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # weights save
    filepath=f"/home/c3238179/COMP4120/weights/weights-improvement-id-{individual_id}-ga-{ga_individual_solution}.hdf5"
    individual_id = individual_id + 1
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # fit the model
    model.fit(X, Y, epochs=15, batch_size = 64, callbacks=callbacks_list)

    start = np.random.randint(0, len(dataX)-1)
    pattern = dataX[start]

    # generate 1000 characters
    textgen = ""
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        textgen += result
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("textgen: " + textgen)

    # evaluate fitness
    from nltk.translate.bleu_score import sentence_bleu
    reference = raw_text.split(' ')
    reference = [reference]
    candidate = textgen.split(' ')
    BLEU_fitness = np.float64(sentence_bleu(reference, candidate))
    print(type(BLEU_fitness))
    print("BLEU: " + str(BLEU_fitness))

    return BLEU_fitness,

# GENETIC ALGORITHM
population_size = 16 # pretty much just based on how many threads are available
num_generations = 10
gene_length = 10

# In case, when you want to maximize accuracy for instance, use 1.0
creator.create('FitnessMax', base.Fitness, weights = (1.0,))
creator.create('Individual', list , fitness = creator.FitnessMax)

pool = multiprocessing.Pool() # multiprocessing
toolbox = base.Toolbox()

toolbox.register('binary', bernoulli.rvs, 0.5)
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.binary, n = gene_length)
toolbox.register('population', tools.initRepeat, list , toolbox.individual)

toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb = 0.6)
toolbox.register('select', tools.selRoulette)
toolbox.register('evaluate', train_evaluate)
toolbox.register("map", pool.map) # multiprocessing

population = toolbox.population(n = population_size)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

r, logbook = algorithms.eaSimple(population, toolbox, cxpb = 0.4, mutpb = 0.1, ngen = num_generations, stats=stats, halloffame=hof, verbose = True)

print ("Logbook:", logbook)
print("r:", r)

print("Hall of Fame:")
for i in len(hof):
    print(" ["+i+"] ")
    print(hof[i])
