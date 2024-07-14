# BEFORE RUNNING THIS FUNCTION:
# Must have executed function MAIN_PREPARE_SAVE_DATA to generate data files.

import pickle
from datetime import datetime
import sys

# Self-defined functions
from load_data import load_data
from read_configuration import read_configuration
from default_configuration import defcon
from make_custom_file_names import model_file_name
from make_custom_file_names import history_file_name
from make_custom_file_names import data_file_name

# custom metrics
from custom_model_elements import my_r_square_metric
from custom_model_elements import my_csi20_metric
from custom_model_elements import my_csi35_metric
from custom_model_elements import my_csi50_metric
from custom_model_elements import my_bias20_metric
from custom_model_elements import my_bias35_metric
from custom_model_elements import my_bias50_metric

# custom loss functions
from custom_model_elements import my_mean_squared_error_noweight
from custom_model_elements import my_mean_squared_error_weighted1
from custom_model_elements import my_mean_squared_error_weighted
from custom_model_elements import my_mean_squared_error_weighted_linear
from custom_model_elements import my_mean_squared_error_weighted_gaussian
from custom_model_elements import my_mean_squared_error_weighted_genexp
from custom_model_elements import my_mean_absolute_error_weighted_genexp
# Note: also need to set this function below the line "##### LOSS FUNCTION"

import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.simplefilter(action='ignore',category=DeprecationWarning)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# required command line argument: my_file_prefix
try:
    my_file_prefix = sys.argv[1]
except IndexError:
    sys.exit('Error: you must supply my_file_prefix as command line argument')
print('my_file_prefix =',my_file_prefix)

config = read_configuration()
print()
print(config)  #because it is so easy to screw-up the config file
print()

################################################################

# machine specific configuration

spath = '..'  #Imme
verbose_fit = 1

try:
    machine = config['machine']
except KeyError:
    try:
        machine = config[my_file_prefix]['machine']
    except KeyError:
        machine = defcon['machine']
print('machine =',machine)

if machine == 'Hera':

    # to avoid GPU out of memory error
    cp = tf.ConfigProto()
    cp.gpu_options.allow_growth = True
    session = tf.Session(config=cp)

    spath = '/scratch1/RDARCH/rda-goesstf/conus2'  #KH on Hera
    verbose_fit = 2  #one progress bar per epoch

################################################################

print('start MAIN_TRAIN_and_SAVE_MODEL=',datetime.now())

##### Load data
try:
    data_suffix = config['data_suffix']
except KeyError:
    try:
        data_suffix = config[my_file_prefix]['data_suffix']
    except KeyError:
        data_suffix = defcon['data_suffix']
print('data_suffix =',data_suffix)

data_file = data_file_name( spath, suffix=data_suffix ) # load file name from file
print('loading data from file =',data_file)
Xdata_train, Ydata_train, Xdata_test, Ydata_test, \
    Lat_train, Lon_train, Lat_test, Lon_test, \
    Xdata_scalar_train, Xdata_scalar_test = load_data( data_file )
nbatches_train,ny,nx,nchans = Xdata_train.shape
nbatches_test,ny,nx,nchans = Xdata_test.shape
print('ny,nx=',ny,nx)
print('nchans=',nchans)
print('nbatches train,test=',nbatches_train,nbatches_test)

if Xdata_scalar_train is None:
    nscalars = 0
    print('no scalars')
else:
    nb_train,ny,ny,nscalars = Xdata_scalar_train.shape
    print('nscalars=',nscalars)

################################################################

##### Load configuration
# Use prefix to add custom names to the file names generated here,
# to make sure we don't overwrite model files, etc.
# Example:  'I1', where I is for Imme and 1 denotes Experiment #1.
print('Configuration:')

### parameter choices for training ###

try:
    NN_string = config['NN_string']
except KeyError:
    try:
        NN_string = config[my_file_prefix]['NN_string']
    except KeyError:
        NN_string = defcon['NN_string']
print('NN_string =',NN_string)

try:
    activ = config['activ']
except KeyError:
    try:
        activ = config[my_file_prefix]['activ']
    except KeyError:
        activ = defcon['activ']
print('activ =',activ)

try:
    activ_last = config['activ_last']
except KeyError:
    try:
        activ_last = config[my_file_prefix]['activ_last']
    except KeyError:
        activ_last = defcon['activ_last']
print('activ_last =',activ_last)

try:
    activ_scalar = config['activ_scalar']
except KeyError:
    try:
        activ_scalar = config[my_file_prefix]['activ_scalar']
    except KeyError:
        activ_scalar = defcon['activ_scalar']
print('activ_scalar =',activ_scalar)

try:
    batch_size = config['batch_size']
except KeyError:
    try:
        batch_size = config[my_file_prefix]['batch_size']
    except KeyError:
        batch_size = int(nbatches_train/defcon['batch_step_size'])
print('batch_size =',batch_size)

try:
    batchnorm = config['batchnorm']
except KeyError:
    try:
        batchnorm = config[my_file_prefix]['batchnorm']
    except KeyError:
        batchnorm = defcon['batchnorm']
print('batchnorm=',batchnorm)

try:
    convfilter = config['convfilter']
except KeyError:
    try:
        convfilter = config[my_file_prefix]['convfilter']
    except KeyError:
        convfilter = defcon['convfilter']
print('convfilter =',convfilter)

try:
    convfilter_last_layer = config['convfilter_last_layer']
except KeyError:
    try:
        convfilter_last_layer = config[my_file_prefix]['convfilter_last_layer']
    except KeyError:
        convfilter_last_layer = defcon['convfilter_last_layer']
print('convfilter_last_layer =',convfilter_last_layer)

try:
    convfilter_scalar = config['convfilter_scalar']
except KeyError:
    try:
        convfilter_scalar = config[my_file_prefix]['convfilter_scalar']
    except KeyError:
        convfilter_scalar = defcon['convfilter_scalar']
print('convfilter_scalar =',convfilter_scalar)

try:
    double_filters = config['double_filters']
except KeyError:
    try:
        double_filters = config[my_file_prefix]['double_filters']
    except KeyError:
        double_filters = defcon['double_filters']
print('double_filters =',double_filters)

try:
    dropout = config['dropout']
except KeyError:
    try:
        dropout = config[my_file_prefix]['dropout']
    except KeyError:
        dropout = defcon['dropout']
print('dropout = ',dropout)

if dropout:
    try:
        dropout_rate = config['dropout_rate']
    except KeyError:
        try:
            dropout_rate = config[my_file_prefix]['dropout_rate']
        except KeyError:
            dropout_rate = defcon['dropout_rate']
    print('dropout_rate = ',dropout_rate)

try:
    kernel_init = config['kernel_init']
except KeyError:
    try:
        kernel_init = config[my_file_prefix]['kernel_init']
    except KeyError:
        kernel_init = defcon['kernel_init']
print('kernel_init =',kernel_init)

##### LOSS FUNCTION #####
try:
    loss = config['loss']
except KeyError:
    try:
        loss = config[my_file_prefix]['loss']
    except KeyError:
        loss = defcon['loss']
print('loss =',loss)
hasweightarg = False
if loss in ['my_mean_squared_error_weighted',\
        'my_mean_squared_error_weighted_linear',\
        'my_mean_squared_error_weighted_gaussian',\
        'my_mean_squared_error_weighted_genexp',\
        'my_mean_absolute_error_weighted_genexp',\
        ]:
    hasweightarg = True
if not hasweightarg:
    if loss == 'my_mean_squared_error_noweight': loss = my_mean_squared_error_noweight
    if loss == 'my_mean_squared_error_weighted1': loss = my_mean_squared_error_weighted1
else:
    try:
        loss_weight = config['loss_weight']
    except KeyError:
        try:
            loss_weight = config[my_file_prefix]['loss_weight']
        except KeyError:
            loss_weight = defcon['loss_weight']
    print('loss_weight=',loss_weight)
    if loss == 'my_mean_squared_error_weighted': loss = my_mean_squared_error_weighted(weight=loss_weight)
    if loss == 'my_mean_squared_error_weighted_linear': loss = my_mean_squared_error_weighted_linear(weight=loss_weight)
    if loss == 'my_mean_squared_error_weighted_gaussian': loss = my_mean_squared_error_weighted_gaussian(weight=loss_weight)
    if loss == 'my_mean_squared_error_weighted_genexp': loss = my_mean_squared_error_weighted_genexp(weight=loss_weight)
    if loss == 'my_mean_absolute_error_weighted_genexp': loss = my_mean_absolute_error_weighted_genexp(weight=loss_weight)
##########

try:
    n_conv_layers_per_decoder_layer = \
        config['n_conv_layers_per_decoder_layer']
except KeyError:
    try:
        n_conv_layers_per_decoder_layer = \
            config[my_file_prefix]['n_conv_layers_per_decoder_layer']
    except KeyError:
        n_conv_layers_per_decoder_layer = \
            defcon['n_conv_layers_per_decoder_layer']
print('n_conv_layers_per_decoder_layer =',n_conv_layers_per_decoder_layer)

try:
    n_conv_layers_per_encoder_layer = \
        config['n_conv_layers_per_encoder_layer']
except KeyError:
    try:
        n_conv_layers_per_encoder_layer = \
            config[my_file_prefix]['n_conv_layers_per_encoder_layer']
    except KeyError:
        n_conv_layers_per_encoder_layer = \
            defcon['n_conv_layers_per_encoder_layer']
print('n_conv_layers_per_encoder_layer =',n_conv_layers_per_encoder_layer)

try:
    n_encoder_decoder_layers = config['n_encoder_decoder_layers']
except KeyError:
    try:
        n_encoder_decoder_layers = config[my_file_prefix]['n_encoder_decoder_layers']
    except KeyError:
        n_encoder_decoder_layers = defcon['n_encoder_decoder_layers']
print('n_encoder_decoder_layers =',n_encoder_decoder_layers)

try:
    n_filters_for_first_layer = config['n_filters_for_first_layer']
except KeyError:
    try:
        n_filters_for_first_layer = config[my_file_prefix]['n_filters_for_first_layer']
    except KeyError:
        n_filters_for_first_layer = defcon['n_filters_for_first_layer']
print('n_filters_for_first_layer =',n_filters_for_first_layer)

try:
    n_filters_last_layer = config['n_filters_last_layer']
except KeyError:
    try:
        n_filters_last_layer = config[my_file_prefix]['n_filters_last_layer']
    except KeyError:
        n_filters_last_layer = defcon['n_filters_last_layer']
print('n_filters_last_layer =',n_filters_last_layer)

try:
    n_filters_scalars = config['n_filters_scalars']
except KeyError:
    try:
        n_filters_scalars = config[my_file_prefix]['n_filters_scalars']
    except KeyError:
        n_filters_scalars = defcon['n_filters_scalars']
print('n_filters_scalars =',n_filters_scalars)

try:
    n_scalar_layers = config['n_scalar_layers']
except KeyError:
    try:
        n_scalar_layers = config[my_file_prefix]['n_scalar_layers']
    except KeyError:
        n_scalar_layers = defcon['n_scalar_layers']
print('n_scalar_layers =',n_scalar_layers)

try:
    nepochs = config['nepochs']
except KeyError:
    try:
        nepochs = config[my_file_prefix]['nepochs']
    except KeyError:
        nepochs = defcon['nepochs']
print('nepochs =',nepochs)

try:
    poolfilter = config['poolfilter']
except KeyError:
    try:
        poolfilter = config[my_file_prefix]['poolfilter']
    except KeyError:
        poolfilter = defcon['poolfilter']
print('poolfilter =',poolfilter)

try:
    upfilter = config['upfilter']
except KeyError:
    try:
        upfilter = config[my_file_prefix]['upfilter']
    except KeyError:
        upfilter = defcon['upfilter']
print('upfilter =',upfilter)

#sys.exit('STOP HERE')

################################################################
##### part below does not change

if NN_string == 'SEQ':
    IS_UNET = False
else:
    IS_UNET = True
print('IS_UNET =',IS_UNET)

layer_format = ['P','CP','CCP','CCCP','CCCCP']
print('encoder layer_format =',layer_format[n_conv_layers_per_encoder_layer])
print('decoder layer_format =',layer_format[n_conv_layers_per_decoder_layer])

optimizer = Adam()
print('optimizer = Adam')

padding = 'same'
print('padding = ',padding)

metrics = [my_r_square_metric,'mean_absolute_error']
metrics.append(my_csi20_metric)
metrics.append(my_csi35_metric)
metrics.append(my_csi50_metric)
metrics.append(my_bias20_metric)
metrics.append(my_bias35_metric)
metrics.append(my_bias50_metric)
if loss != 'mean_squared_error':
    metrics.append('mean_squared_error')

# Tell the user about architecture
#if IS_UNET:
#    print('\nArchitecture: Unet')
#else:
#    print('\nArchitecture: Standard sequential')
#print('Blocks: ' + repr(n_encoder_decoder_layers))
#print('Epochs: ' + repr(nepochs))
#print('Batch size: ' + repr(batch_size))

##### Get file names
modelfile = model_file_name(spath, IS_UNET, my_file_prefix, \
    n_encoder_decoder_layers, nepochs )
historyfile = history_file_name( spath, IS_UNET, my_file_prefix, \
    n_encoder_decoder_layers, nepochs )

print('\nResults will be stored here:')
print( '   Model file: ' + modelfile )
print( '   History file: ' + historyfile )

################################################################
# Define model: encoder-decoder structure

stime = datetime.now()
print('\nstart',stime)

n_filters = n_filters_for_first_layer

input_layer = Input(shape=(ny, nx, nchans))
x = input_layer
if IS_UNET:
    skip = []  #UNET has skip connections

### contracting path (encoder layers)
for i_encode_decoder_layer in range( n_encoder_decoder_layers ):
    print('Add encoder layer #' + repr(i_encode_decoder_layer) )
    if IS_UNET:
        skip.append(x)  # push current x on top of stack

    for i in range(n_conv_layers_per_encoder_layer): #add conv layer
        x = Conv2D(n_filters,convfilter,activation=activ,\
            padding=padding,kernel_initializer=kernel_init)(x)
        if batchnorm:
            x = BatchNormalization()(x)
    x = MaxPooling2D(poolfilter,padding=padding)(x)
    if dropout:
        x = Dropout(dropout_rate)(x)
    if double_filters:
        n_filters = n_filters * 2 # double for NEXT layer

### expanding path (decoder layers)
for i_encode_decoder_layer in range( n_encoder_decoder_layers ):
    print('Add decoder layer #' + repr(i_encode_decoder_layer) )

    # This was moved up to make endcoder and decoder symmetric.
    if double_filters:
        n_filters = n_filters // 2 # halve for NEXT layer
    
    for i in range(n_conv_layers_per_decoder_layer): #add conv layer
        # Switched from Conv2DTranspose to Conv2D.  Same functionality, but easier to visualize filters.
        x = Conv2D(n_filters,convfilter,activation=activ,\
            padding=padding,kernel_initializer=kernel_init)(x)
        if batchnorm:
            x = BatchNormalization()(x)
    x = UpSampling2D(upfilter)(x)
    if IS_UNET:
        x = Concatenate()([x,skip.pop()]) # pop top element
    if dropout:
        x = Dropout(dropout_rate)(x)
    
if IS_UNET:
    # One additional (3x3) conv layer to properly incorporate newly
    # added channels at previous concatenate step
    x = Conv2D(n_filters,convfilter,activation=activ,\
        padding=padding,kernel_initializer=kernel_init)(x)

# Now add the scalars as additional channels - one channel for each scalar:
if nscalars > 0:
    input_scalars = Input(shape=(ny,nx,nscalars))
    x = Concatenate()([x,input_scalars])

    # scalar layers to incorporate scalar information
    for i in range(n_scalar_layers):
        x = Conv2D(n_filters_scalars,convfilter_scalar,activation=activ_scalar,\
            padding=padding,kernel_initializer=kernel_init)(x)
        if batchnorm:
            x = BatchNormalization()(x)
        if dropout:
            x = Dropout(dropout_rate)(x)

# last layer: 2D convolution with (1x1) just to merge the channels
x = Conv2D(n_filters_last_layer,convfilter_last_layer,\
    activation=activ_last,padding=padding,\
    kernel_initializer=kernel_init)(x)
x = Reshape((ny,nx))(x)
if nscalars == 0:
    model = Model(inputs=input_layer, outputs=x)
else:
    model = Model(inputs=[input_layer,input_scalars], outputs=x)

if IS_UNET:
    print('Unet !!!!')
else:
    print('SEQ !!!!')

# Architecture definition complete

print('\n')
print(model.summary())  # print model architecture

#    print('# encoder/decoder layers = ' + repr(n_encoder_decoder_layers))
#    print('# epochs =' + repr(nepochs))
#    print('batch size =' + repr(batch_size))

########### TRAIN MODEL ###########
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

if nscalars == 0:
    history = model.fit(Xdata_train,Ydata_train,epochs=nepochs,\
        batch_size=batch_size,shuffle=True,\
        validation_data=(Xdata_test,Ydata_test), \
        verbose=verbose_fit)
else:
    history = model.fit([Xdata_train,Xdata_scalar_train],Ydata_train,epochs=nepochs,\
        batch_size=batch_size,shuffle=True,\
        validation_data=([Xdata_test,Xdata_scalar_test],Ydata_test), \
        verbose=verbose_fit)

# Time statistics
etime = datetime.now()
print('end',etime)
print('Time ellapsed for training',(etime-stime).total_seconds(),\
    'seconds\n')

########### SAVE MODEL ###########
print('Writing model to file: ' + modelfile + '\n')
model.save(modelfile)

########### Save training history ###########
print('Writing history to file: ' + historyfile + '\n')
with open(historyfile, 'wb') as f:
    pickle.dump({'history':history.history, 'epoch':history.epoch}, f)

######################################################################

print('TRAINING:  Done!')
print('To inspect results, run function '\
    'MAIN_POST_PROCESSING or function MAIN_VISUALIZATION\n')

print('end MAIN_TRAIN_and_SAVE_MODEL=',datetime.now())
