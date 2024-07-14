import numpy as np, pandas as pd, matplotlib.pyplot as plt, sys

from time import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from neuralNet import NeuralNet
from keras import layers
from keras import backend as K

#sys.path.insert(0, '/its/home/mb756/python/tools21cm')
#from clump_functions import ReadTensor, PrintTimeElapsed

np.random.seed(0)
rho_c0 = 9.204552e-30

def ReadTensor(filename, bits=32, order='C', dimensions=3):
    ''' Read a binary file with three inital integers (a cbin file).

    Parameters:
        * filename (string): the filename to read from
        * bits = 32 (integer): the number of bits in the file
        * order = 'C' (string): the ordering of the data. Can be 'C'
            for C style ordering, or 'F' for fortran style.
        * dimensions (int): the number of dimensions of the data (default:3)

    Returns:
        The data as a three dimensional numpy array.
    '''

    assert(bits == 32 or bits == 64)

    f = open(filename)

    temp_mesh = np.fromfile(f, count=dimensions, dtype='int32')

    datatype = np.float32 if bits == 32 else np.float64
    data = np.fromfile(f, dtype=datatype, count=np.prod(temp_mesh))
    data = data.reshape(temp_mesh, order=order)
    return data

def R2(y_true, y_pred):
    ''' R^2 (coefficient of determination) regression score function. To avoid NaN, added 1e-8 to denominator ''' 
    SS_num =  K.sum(K.square(y_true-y_pred)) 
    SS_den = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - SS_num/(SS_den + K.epsilon())


def main():
    path = './data/'
    filename = 'reionNNdataset_47Mpc_64000x68_5.dat'
    data = pd.DataFrame(data=ReadTensor(path+filename, dimensions=2), columns=['z', 'nigm', 'M9', 'irate', 'xi']).set_index('z')


    # Transform the data (logarithm is sometime easyer for NN to learn)
    data = data.drop(['xi'], axis=1)
    #data = data.drop(['M89'], axis=1)
    data.loc[:, 'nigm'] = np.log10(data.loc[:, 'nigm']/rho_c0)
    idx = np.nonzero(data.loc[:, 'irate'].values)
    nonzeromin = np.min(data.loc[:, 'irate'].values[idx])
    data.loc[:, 'irate'] = np.log10(data.loc[:, 'irate']+nonzeromin)

    print("Split data into training/testing sets, and input/output sets")
    training, testing = train_test_split(data, test_size=0.2)
    train_x, test_x = training.iloc[:, :-1], testing.iloc[:, :-1]
    train_y, test_y = training.loc[:, 'irate'], testing.loc[:, 'irate']
     
    #Normalize x-data to [0,1]
    scale_x = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(train_x)
    train_x, test_x = scale_x.transform(train_x), scale_x.transform(test_x)
    scale_y = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(train_y.values.reshape(-1,1))
    train_y, test_y = scale_y.transform(train_y.values.reshape(-1,1)), scale_y.transform(test_y.values.reshape(-1,1))

    print("Set the hyperparameters")
    network = NeuralNet(INPUT_DIM     = train_x.shape[1], OUTPUT_DIM	= 1,
                        HIDDEN_DIM_1  = 150,              HIDDEN_DIM_2	= 100,
                        BATCH_SIZE    = 32,               LOSS		    = 'mse',
                        LEARNING_RATE = 0.01,             EPOCHS	    = 10,
                        DROPOUT       = 0.2,              METRICS	    = ['mae', 'mse', R2],
                        DATA_USED     = filename,         VARIABLES	    = data.columns.values,
                        GPU           = None,             RESUME_EPOCH  = 0,
                        RESUME_PATH   = None)
    
    np.savetxt(network.path+'scale_params.txt', [np.min(data.loc[:, 'irate'].values), np.max(data.loc[:, 'irate'])], header=' Normalization of target X is donei:\n X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))\n X_scaled = X_std * (1 - 0) + 0')
    
    # Builds/compiles the network
    network.compile()       # TODO: CHECK THAT GPU ALLOCATION IS CORRECT

    # Functions applied at given stages of network training
    network.add_callbacks(reduce_lr=True, checkpoints=10)

    # Train the network, Test set used for validation")
    network.fit(train_x, train_y, test_x, test_y)

    # Uses trained network to predict outputs/loss for validation set
    network.predict(test_x ,test_y)
    
    #Scale prediction to original format
    network.predictions      = np.power(10, scale_y.inverse_transform(network.predictions))-nonzeromin
    network.expected_outputs = np.power(10, scale_y.inverse_transform(network.expected_outputs))-nonzeromin
    
    print('Testing Loss =  ', network.accuracy[0])
    
    network.save_best_network()
    network.save_outputs()
    network.plot_predictions(save_fig='yes')
    network.plot_loss(save_fig='yes')
    

t_start = time()
if __name__ == '__main__':
    main()
