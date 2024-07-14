import sys, pickle, os, gzip, time
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import keras, warnings
from tensorflow.compat.v1 import InteractiveSession
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = InteractiveSession(config=config)
warnings.filterwarnings(action='once')

print('Python version:{}\n'.format(sys.version))
print('TF version:{}, Keras version:{}\n'.format(tf.__version__, keras.__version__))

t1 = time.time()
############################# LOAD THE DATA AND ONE HOT VECTORIZE THE LABELS ###########################################

filename = 'data/mnist.pkl.gz'
filehandle = gzip.open(filename, 'rb')
train_data, val_data, test_data = pickle.load(filehandle, encoding='latin1')
#train_data, val_data, test_data = pickle.load(filehandle) #<--- for python2.7
#train_data, val_data, test_data = pickle.load(filehandle)
filehandle.close()
train_x, train_y = train_data
print('train_x.shape:{} and train_y.shape:{}'.format(train_x.shape, train_y.shape))
val_x, val_y = val_data
print('val_x.shape:{} and val_y.shape:{}'.format(val_x.shape, val_y.shape))
print('images are 28*28 = 784 vectors')
#print(train_x[0]) # images are float32, normalized, and 784 vectors
print('')
# combine train and validation data as Keras will split it internally
train_x = np.concatenate([train_x, val_x], axis=0)
train_y = np.concatenate([train_y, val_y], axis=0)
print('train_x.shape:{}'.format(train_x.shape))
print('train_y.shape:{}'.format(train_y.shape))
print('')
print('train_x[0].shape:{}'.format(train_x.shape[0]))
print('train_x[1].shape:{}'.format(train_x.shape[1]))
print('')
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
print('train_x.shape:{}'.format(train_x.shape))
print('train_y.shape:{}'.format(train_y.shape))
print('train_x[59999].shape: {}'.format(train_x[59999].shape))
#print('train_x[0]: {}'.format(train_x[0]))
print('')
test_x, test_y = test_data
print('test_x.shape:{}'.format(test_x.shape))
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
print('test_x.shape:{}'.format(test_x.shape))
print('test_y.shape:{}'.format(test_y.shape))
#train_x.shape[0]


train_y = to_categorical(train_y)
print('train_y.shape:{}'.format(train_y.shape))

test_y = to_categorical(test_y)
print('test_y.shape:{}'.format(test_y.shape))

##################### SETUP THE NEURAL NETWORK PARAMETES ###########################################################################

eta = 0.0005
val_frac = 0.1 #Fraction of training data to be used for validation
# FIRST CONVOLUTION LAYER
nC1_kernels = 32 ##number of kernels in the first convolutional layer
C1_kernel_shape = (5, 5) ## size of the kernel in the first convolutional layer (5,5)
C1_stride = (1,1) ## stride of the convolution 1 pixel right and 1 pixel down
C1_activation = 'relu' ## activation function of the C1 neurons
P1_kernel_shape = (2, 2) ## Size of the pooling window (2,2)
P1_stride = 2 ## stride of the pooling window

# SECOND CONVOLUTION LAYER
nC2_kernels = 64
C2_kernel_shape = (3, 3)  # kernel weight is really 3x3x32
C2_stride = 1
C2_activation = 'relu'
P2_kernel_shape = (2, 2)
P2_stride = 2

# FULLY CONNECTED LAYER
n_dense = 200
dense_activation = 'relu'

# FINAL LAYER
last_activation = 'softmax'
cost_function = 'categorical_crossentropy'
n_out = 10
optimizer = 'adam' #'sgd'



########## SETUP THE NEURAL NETWORK #############################################################################
model = tf.keras.Sequential()
#FIRST CONVOLUTION LAYER
model.add(tf.keras.layers.Conv2D(nC1_kernels, C1_kernel_shape, C1_stride, activation=C1_activation,
                                input_shape=(28, 28, 1)))
# The 32 maps are 24x24   # 28-5+1 = 24
model.add(tf.keras.layers.MaxPooling2D(P1_kernel_shape, P1_stride))
# The 32 pooled maps are 12x12
#SECOND CONVOLUTION LAYER
model.add(tf.keras.layers.Conv2D(nC2_kernels, C2_kernel_shape, C2_stride, activation=C2_activation))
# The 64 maps are 10x10   (12-3+1 = 10)
model.add(tf.keras.layers.MaxPooling2D(P2_kernel_shape, P2_stride))
# The 64 pooled maps are 5x5
model.add(tf.keras.layers.Flatten())
# This makes the 64 pooled 5x5 neuronal maps into a single 25*64 = 1600 neuron layer
#FULLY CONNECTED LAYERS
model.add(tf.keras.layers.Dense(n_dense, activation=dense_activation))
model.add(tf.keras.layers.Dense(n_out, activation=last_activation))

#COMPILE THE MODEL
if(optimizer=='adam'):
    optim = tf.keras.optimizers.Adam(lr=eta)
else:
    optim = tf.keras.optimizers.SGD(lr=eta)

model.compile(optimizer=optim, loss=cost_function, metrics=['accuracy'])
print(model.summary())

#CALLBACK TO KERAS TO SAVE BEST MODEL WEIGHTS
best_weights='outputs/cnn_weights_best_'+str(os.getpid())+'_.hdf5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(best_weights, monitor='val_acc', verbose=1, save_best_only=True,
                                                mode='max')


######################## TRAIN THE CNN ###########################################################################
mini_batch_size = 32
history = model.fit(train_x,train_y, epochs=3, batch_size=mini_batch_size, callbacks=[checkpoint],
                   validation_split=val_frac)
# val_frac is fraction of the training data taken to be validation data

##########################TEST THE MODEL AT THE END#########################################################################
print('Test result of the final epoch\n')
model.evaluate(test_x, test_y, batch_size=len(test_x))


########################### TEST THE MODEL AT THE BEST VALIDATION ECPOCH ############################################
print('Test accuracy of the epoch that resulted  in the best validation accuracy\n')
model.load_weights(best_weights)
model.compile(optimizer=optim, loss=cost_function, metrics=['accuracy'])
model.evaluate(test_x, test_y, batch_size=len(test_x))

################################ SAVE THE MODEL HISTORY ####################################
picklefile = 'outputs/cnn_model_history_'+str(os.getpid())+'_.pkl'
output = open(picklefile,'wb')
pickle.dump(history.history, output)
output.close()
print('pickle file written to:{}'.format(picklefile))

print('Total time taken:{}'.format(time.time()-t1))
######################## LOAD THE HISTORY (USE ANOTHER .PY FILE TO PLOT ON YOUR LOCAL MACHINE) ###################
#picklefile = open('cnn_model_history.pkl', 'rb') 
#loaded_history = pickle.load(picklefile)
#picklefile.close()



