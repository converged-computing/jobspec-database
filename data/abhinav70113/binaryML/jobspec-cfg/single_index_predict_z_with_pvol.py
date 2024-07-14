import numpy as np
import tensorflow as tf
import sys
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import os
import glob
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import ast
from tensorflow.keras.layers import MultiHeadAttention, Attention
import json
import time
import pandas as pd
import argparse
from functools import partial
parser = argparse.ArgumentParser()

def myexecute(cmd):
    if 'echo' not in cmd[:20]:
        os.system("echo '%s'"%cmd)
    os.system(cmd)

parser.add_argument('--job_id', type=str, default='0')
parser.add_argument('--model_type', type=str, default='cnn')
parser.add_argument('--trial_index', type=int, default=985)
parser.add_argument('--max_seconds', type=int, default=13500)
parser.add_argument('--snr_group', type=int, default=-1) # -1 means all snr groups otherwise vary it from 0 to 4
parser.add_argument('--previous_job_id', type=str)

args = parser.parse_args()

job_id = args.job_id
model_type = args.model_type
trial_index = args.trial_index
max_seconds = args.max_seconds
snr_group = args.snr_group

run = 'runBC'

cur_dir = f'/tmp/Abhinav_DATA{job_id}/'
root_dir = '/hercules/scratch/atya/BinaryML/'
myexecute(f'mkdir -p {cur_dir}raw_data/{run}/')
myexecute(f'mkdir -p {cur_dir}models/')
resume_training = False

if args.previous_job_id:
    previous_job_id = args.previous_job_id
    resume_training_previous_job_id = previous_job_id
    best_model_name_pattern = f'{root_dir}models/*{resume_training_previous_job_id}*checkpoint.h5'
    print(best_model_name_pattern)
    best_models = glob.glob(best_model_name_pattern)
    print(best_models)
    if len(best_models) == 0:
        raise ValueError('No previous model found')
    elif len(best_models) > 1:
        raise ValueError('Need a unique job id to resume training')
    else:
        file_name_checkpoint_root = best_models[0]
        myexecute(f'echo "Found previous model: {file_name_checkpoint_root}"')
        resume_training = True
        myexecute(f'rsync -Pav -q {file_name_checkpoint_root} {cur_dir}models/')
        best_model_name_pattern = f'{cur_dir}models/*{resume_training_previous_job_id}*checkpoint.h5'
        best_models = glob.glob(best_model_name_pattern)
        file_name_checkpoint = best_models[0]
        best_model_name_resume = file_name_checkpoint[:-len('checkpoint.h5')]

if model_type == 'cnn':
    #list_of_dicts = json.load(open(f'/hercules/scratch/atya/BinaryML/hyperparameter_tuning_high_snr/cnn_f_fine/list_of_dicts.json','r'))
    list_of_dicts = json.load(open(f'/hercules/scratch/atya/BinaryML/hyperparameter_tuning_runBC/cnn/list_of_dicts.json','r'))

elif model_type == 'attention':
    #list_of_dicts = json.load(open(f'/hercules/scratch/atya/BinaryML/hyperparameter_tuning_high_snr/attention_z_fine/list_of_dicts.json','r'))
    list_of_dicts = json.load(open(f'/hercules/scratch/atya/BinaryML/hyperparameter_tuning_runBC/attention_z/list_of_dicts.json','r'))


#model_type = 'LSTM'
param_dict = list_of_dicts[trial_index]
input_shape = param_dict['input_shape']
batch_size = param_dict['batch_size']
epochs = 20000
patience = 200
best_model_name = f'{cur_dir}models/low_snr_predict_{model_type}_{job_id}_'


myexecute(f'echo \"\n\n\n\n############################################################################## \n\n\n \
          Adam learning rate 0.01 and loss funciton is the ratio of the powers added to the mse \n \
          Comment: Low snr Binary simulations predictor: \n \
          Best model is under:{best_model_name} \n \
          model_type: {model_type} \n \
          trial_index: {trial_index} \n \
          snr_group: {snr_group} \n \
           \n\n\n ############################################################################## \n\n\n \"')



#files to sync
files1 = glob.glob(f'{root_dir}raw_data/{run}/*chunk.npy')
files1.extend(glob.glob(f'{root_dir}raw_data/{run}/*indices.npy'))
files1.extend(glob.glob(f'{root_dir}raw_data/{run}/*rslices.npy'))
for file in files1:
    myexecute(f'rsync -Pav -q {file} {cur_dir}raw_data/{run}/')


X_train = np.load(cur_dir + f'raw_data/{run}/train_data_chunk.npy').astype(np.float64)
X_test = np.load(cur_dir + f'raw_data/{run}/test_data_chunk.npy').astype(np.float64)
X_val = np.load(cur_dir + f'raw_data/{run}/val_data_chunk.npy').astype(np.float64)
X_train = X_train/np.max(X_train,axis=1)[:,None]
X_test = X_test/np.max(X_test,axis=1)[:,None]
X_val = X_val/np.max(X_val,axis=1)[:,None]

# X_train = X_train[:1200]
# X_test = X_test[:400]
# X_val = X_val[:400]

Y_train = np.load(cur_dir + f'raw_data/{run}/train_labels_chunk.npy').astype(np.float64)
Y_test = np.load(cur_dir + f'raw_data/{run}/test_labels_chunk.npy').astype(np.float64)
Y_val = np.load(cur_dir + f'raw_data/{run}/val_labels_chunk.npy').astype(np.float64)

# X_train = np.concatenate((X_train, Y_train[:, 0].reshape(-1,1)), axis=1)
# X_test = np.concatenate((X_test,Y_test[:,0].reshape(-1,1)),axis=1)
# X_val = np.concatenate((X_val,Y_val[:,0].reshape(-1,1)),axis=1)

# Y_train = Y_train[:1200]
# Y_test = Y_test[:400]
# Y_val = Y_val[:400]
test_rslices = np.load(cur_dir + f'raw_data/{run}/test_rslices.npy')
train_rslices = np.load(cur_dir + f'raw_data/{run}/train_rslices.npy')
val_rslices = np.load(cur_dir + f'raw_data/{run}/val_rslices.npy')



if model_type == 'cnn':
    Y_train = np.abs(Y_train[:,0])
    Y_test = np.abs(Y_test[:,0])
    Y_val = np.abs(Y_val[:,0])
elif model_type == 'attention':
    Y_train = 2*np.abs(Y_train[:,1])
    Y_test = 2*np.abs(Y_test[:,1])
    Y_val = 2*np.abs(Y_val[:,1])

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1) 

Y_train = (Y_train.reshape(Y_train.shape[0],1))
Y_val = (Y_val.reshape(Y_val.shape[0],1))
Y_test = (Y_test.reshape(Y_test.shape[0],1))

train_indices = np.load(cur_dir + f'raw_data/{run}/train_indices.npy')
test_indices = np.load(cur_dir + f'raw_data/{run}/test_indices.npy')
val_indices = np.load(cur_dir + f'raw_data/{run}/val_indices.npy')



def find_indices(small_list, big_list):
    indices = []

    for item in small_list:
        if item in big_list:
            index = np.where(big_list == item)[0]
            indices.append(index[0])

    return indices

if snr_group != -1:

    labels_df = pd.read_csv(root_dir + f'meta_data/labels_{run}.csv')
    #snr_range = labels_df['snr'].values
    snr_range = labels_df['z'].values
    #snr_bins = np.array([0.005 , 0.068, 0.126, 0.184, 0.242, 0.3  ])
    #snr_bins = np.array([0 , 5, 30])
    #snr_bins = np.array([0.005  , 0.00625, 0.0075 , 0.00875, 0.01   ])
    #snr_bins = np.array([ 0.        , 16.66666667, 33.33333333, 50.        ])
    snr_bins = np.array([-100,0,100])
    #snr_bins = np.array([ 0.        ,  5.55555556, 11.11111111, 16.66666667, 22.22222222,
    #   27.77777778, 33.33333333, 38.88888889, 44.44444444, 50.        ])

    train_indices_group = labels_df['# ind'][(snr_range > snr_bins[snr_group]) & (snr_range < snr_bins[snr_group + 1]) & (labels_df['status'] == 'train')].values
    test_indices_group = labels_df['# ind'][(snr_range > snr_bins[snr_group]) & (snr_range < snr_bins[snr_group + 1]) & (labels_df['status'] == 'test')].values
    val_indices_group = labels_df['# ind'][(snr_range > snr_bins[snr_group]) & (snr_range < snr_bins[snr_group + 1]) & (labels_df['status'] == 'val')].values

    train_indices_small = find_indices(train_indices_group, train_indices)
    test_indices_small = find_indices(test_indices_group, test_indices)
    val_indices_small = find_indices(val_indices_group, val_indices)

    X_train = X_train[train_indices_small]
    X_test = X_test[test_indices_small]
    X_val = X_val[val_indices_small]

    Y_train = Y_train[train_indices_small]
    Y_test = Y_test[test_indices_small]
    Y_val = Y_val[val_indices_small]

    test_rslices = test_rslices[test_indices_small]
    train_rslices = train_rslices[train_indices_small]
    val_rslices = val_rslices[val_indices_small]

Y = np.concatenate([Y_val,Y_test,Y_train], axis=0)

max_val_slice = np.max(val_rslices,axis=1)
max_test_slice = np.max(test_rslices,axis=1)
max_train_slice = np.max(train_rslices,axis=1)
# concatenating rslices
rslices = np.concatenate([test_rslices, train_rslices, val_rslices], axis=0)

# concatenating max_r_slices
max_r_slices = np.concatenate([max_val_slice, max_test_slice, max_train_slice], axis=0)

myexecute(f'echo "Training data shape: {X_train.shape}"')
myexecute(f'echo "Test data shape: {X_test.shape}"')
myexecute(f'echo "Validation data shape: {X_val.shape}"')
myexecute(f'echo "Training labels shape: {Y_train.shape}"')
myexecute(f'echo "Test labels shape: {Y_test.shape}"')
myexecute(f'echo "Validation labels shape: {Y_val.shape}"')
myexecute(f'echo "Training rslices shape: {train_rslices.shape}"')
myexecute(f'echo "Test rslices shape: {test_rslices.shape}"')
myexecute(f'echo "Validation rslices shape: {val_rslices.shape}"')
print(Y_train[:50])

# Set up the TimeLimitCallback callback
class TimeLimitCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_seconds):
        self.max_seconds = max_seconds
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self.start_time > self.max_seconds:
            self.model.stop_training = True
            print("\nReached time limit. Stopping training...")
    
    def on_train_end(self, logs=None):
        if time.time() - self.start_time > self.max_seconds:
            self.model.stop_training = True
            myexecute('echo "Stopped training due to time limit"')

# Create data generators
def create_dataset(X, y, batch_size, shuffle):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def period_accuracy(Y_test, Y_pred):
    """
    Computes the percentage of predicted periods that are within 0.5% of the true periods.
    """
    accuracies = []
    Y_test = tf.cast(Y_test, tf.float32)
    Y_pred = tf.cast(Y_pred, tf.float32)
    shape = Y_test.shape[1]
    for i in range(shape):
        rat = (Y_test[:,i]+ 1e-10)/(Y_pred[:,i]+ 1e-10)
        rat_bool = tf.logical_and(tf.less(rat, 1.005), tf.greater(rat, 0.995))
        accuracy = tf.reduce_mean(tf.cast(rat_bool, tf.float32)) * 100
        accuracies.append(accuracy)
    return [acc.numpy() for acc in accuracies]


def median_percent_deviation(Y_test, Y_pred):
    """
    Computes the median percent deviation from the predicted period.
    """
    Y_test = tf.cast(Y_test, tf.float32)
    Y_pred = tf.cast(Y_pred, tf.float32)
    median_deviations = []
    shape = Y_test.shape[1]
    for i in range(shape):
        
        Y_test_i = Y_test[:,i]
        Y_pred_i = Y_pred[:,i]

        percent_deviation = tf.abs((Y_test_i - Y_pred_i) / (Y_test_i + 1e-10)) * 100
        median_deviation = tfp.stats.percentile(percent_deviation, q=50)
        median_deviations.append(median_deviation)

    return [med.numpy() for med in median_deviations]


def attention_model(param_dict):

    num_cnn_layers = param_dict['num_cnn_layers']
        
    input_shape = param_dict['input_shape']

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    batch_normalization = param_dict['batch_normalization'] #take true or false as input
    #conveting the strings to lists
    conv1d_filters = ast.literal_eval(param_dict['conv1d_filters'])
    conv1d_kernel_size = ast.literal_eval(param_dict['conv1d_kernel_size'])
    dense_units = ast.literal_eval(param_dict['deep_layer_size'])

    for i in range(num_cnn_layers):
            x = layers.Conv1D(filters=conv1d_filters[i],
                              kernel_size=conv1d_kernel_size[i],
                              padding='same',)(x)
            if batch_normalization:
                x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

    attention_type = param_dict['attention_type']
    num_attention_layers = param_dict['num_attention_layers']
    if attention_type == 'multi_head':
        num_attention_heads = param_dict['num_attention_heads']
        num_key_dims = param_dict['num_key_dims']
        for i in range(num_attention_layers):
            x = MultiHeadAttention(num_heads=num_attention_heads, key_dim=num_key_dims)(x,x)
    elif attention_type == 'simple':
        for i in range(num_attention_layers):
            x = layers.Attention()([x,x])
    #x = Dropout(param_dict['dropout_rate'])(x)
    x = layers.Flatten()(x)
    
    num_deep_layers = param_dict['num_deep_layers']
    for j in range(num_deep_layers):
        x = layers.Dense(dense_units[j],activation='relu')(x)
    
    final_outputs = layers.Dense(1, activation='relu')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=final_outputs)


    return model

def cnn_model(param_dict):

    num_cnn_layers = param_dict['num_cnn_layers']
        
    input_shape = param_dict['input_shape']

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    dilation = param_dict['dilation'] #take true or false as input
    batch_normalization = param_dict['batch_normalization'] #take true or false as input
    if dilation:
        dilation_rate_size = ast.literal_eval(param_dict['dilation_rate_size'])
    #conveting the strings to lists
    conv1d_filters = ast.literal_eval(param_dict['conv1d_filters'])
    conv1d_kernel_size = ast.literal_eval(param_dict['conv1d_kernel_size'])
    dense_units = ast.literal_eval(param_dict['deep_layer_size'])

    if dilation:
        for i in range(num_cnn_layers):
            x = layers.Conv1D(filters=conv1d_filters[i],
                    kernel_size=conv1d_kernel_size[i],
                    padding=param_dict['padding'],
                    dilation_rate=dilation_rate_size[i])(x)
            if batch_normalization:
                x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

    else:
        for i in range(num_cnn_layers):
            x = layers.Conv1D(filters=conv1d_filters[i],
                              kernel_size=conv1d_kernel_size[i],
                              padding=param_dict['padding'],)(x)
            if batch_normalization:
                x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)

    #x = Dropout(param_dict['dropout_rate'])(x)
    x = layers.Flatten()(x)
    
    num_deep_layers = param_dict['num_deep_layers']
    for j in range(num_deep_layers):
        x = layers.Dense(dense_units[j],activation='relu')(x)
    
    final_outputs = layers.Dense(1, activation='relu')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=final_outputs)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=param_dict['initial_learning_rate'],
    decay_steps=10000,
    decay_rate=param_dict['decay_rate'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optimizer=optimizer,
              loss='mse', metrics=['mse', 'mae'])
    return model


def current_model(model_name):
    if model_name == 'attention':
        return attention_model(param_dict)
    elif model_name == 'cnn':
        return cnn_model(param_dict)
    else:
        raise ValueError('Model name not found')

model = current_model(model_name = model_type)

# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
# initial_learning_rate=param_dict['initial_learning_rate'],
# decay_steps=10000,
# decay_rate=param_dict['decay_rate'])
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

def find_z_ind(z_array, z_value):
    return tf.argmin(tf.abs(z_array - z_value), axis=-1)

def loss_pvol_calc(index_true, index_pred, rslices, max_r_slice):
    rslice_value = tf.gather_nd(rslices, [index_true, index_pred])
    max_r_value = tf.gather(max_r_slice, index_true)
    result = 1 - tf.divide(rslice_value, max_r_value)
    #result = tf.abs(rslice_value - max_r_value)
    result = tf.maximum(result, 0)
    return tf.cast(result,tf.float32)

def loss_function(y_true, y_pred, rslices, y_true_array, zarray, max_r_slice):
    index_true_list = tf.map_fn(lambda y: find_z_ind(y_true_array, y), y_true, dtype=tf.int64)
    index_pred_list = tf.map_fn(lambda y: find_z_ind(zarray, y), y_pred, dtype=tf.int64)
    
    # Reshape tensors to ensure they have the same shape
    index_true_list = tf.reshape(index_true_list, shape=[-1, 1])
    index_pred_list = tf.reshape(index_pred_list, shape=[-1, 1])
    
    #print(index_true_list.shape)
    #print(index_pred_list.shape)

    indices = tf.concat([index_true_list, index_pred_list], axis=-1)
    loss2 = tf.map_fn(lambda ind: loss_pvol_calc(ind[0], ind[1], rslices, max_r_slice), indices, dtype=tf.float32)
    loss1 = tf.square(y_true - y_pred)
    loss2 = tf.reshape(tf.cast(loss2, tf.float32),[-1,])
    loss1 = tf.cast(loss1, tf.float32)
    loss2_factors = (200 - tf.cast(y_true, tf.float32))**2
    loss = loss1 + tf.multiply(loss2_factors,loss2)
    #print(loss1.shape)
    #print(loss2.shape)
    return loss

rslices = tf.convert_to_tensor(rslices, dtype=tf.float32)
zarray = tf.convert_to_tensor(np.arange(200.0, -200.0, -8.0/256), dtype=tf.float32)
max_r_slices = tf.convert_to_tensor(max_r_slices, dtype=tf.float32)
y_true_array = tf.convert_to_tensor(Y.reshape(Y.shape[0],), dtype=tf.float32)
# Create a version of your loss function with the extra arguments filled in
custom_loss = partial(loss_function, rslices=rslices, y_true_array=y_true_array, zarray=zarray, max_r_slice=max_r_slices)

# Ensure the new function only requires two arguments: y_true and y_pred
custom_loss.__name__ = 'custom_loss'

model.compile(optimizer=optimizer,
            loss=custom_loss, metrics=['mse', 'mae'])
# Print model summary
model.summary()

# Evaluate before loading weights
initial_loss = model.evaluate(X_val, Y_val, verbose=0)

checkpoint_filepath = best_model_name+'checkpoint.h5'
 
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

time_limit_callback = TimeLimitCallback(max_seconds=max_seconds)
print(resume_training)

if resume_training:
        model.load_weights(file_name_checkpoint)  
        print('Resuming training from checkpoint')
        # Evaluate after loading weights
        loaded_loss = model.evaluate(X_val, Y_val, verbose=0)
        print(f'Initial loss: {initial_loss}')
        print(f'Loss after loading weights: {loaded_loss}')
        print('Setting the best validation loss from previous training')


             
train_generator = create_dataset(X_train, Y_train, batch_size=batch_size, shuffle=True)
val_generator = create_dataset(X_val, Y_val, batch_size=batch_size, shuffle=False)

# Define early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min', restore_best_weights=True)

# Train the model
model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=[early_stop, checkpoint_callback,time_limit_callback], verbose=2)

model.save(best_model_name+'model.h5')

# Load the best model after training
#best_model = tf.keras.models.load_model(best_model_name+'checkpoint.h5', custom_objects={'custom_tanh_activation': custom_tanh_activation,'weighted_mse_loss': weighted_mse_loss, 'custom_sigmoid_activation': custom_sigmoid_activation,
#                                                                                         'weighted_mse_loss_wrapper' : get_weighted_mse_loss(default_hyperparameters['power'],default_hyperparameters['factor_mse'])})

best_model = model
# Test model on X_test
Y_pred = best_model.predict(X_test)
Y_pred = np.reshape(Y_pred, Y_test.shape)
# Y_pred_array = np.zeros_like(Y_test)
# Y_pred_array[:,0] = Y_pred[0].reshape(-1)
# Y_pred_array[:,1] = Y_pred[1].reshape(-1)
# Y_pred = Y_pred_array

accuracy = period_accuracy(Y_test, Y_pred)
median = median_percent_deviation(Y_test, Y_pred)


myexecute(f'echo \"\n\n\n\n############################################################################## \n\n\n \
          The accuracy of the model is:{accuracy} \n\
          The median absolute error of the model is:{median} \n\
          \n\n\n ############################################################################## \n\n\n \"')

myexecute(f'rsync -q -Pav {os.path.join(cur_dir, best_model_name)}checkpoint.h5 {root_dir}models/ ')
myexecute(f'rsync -q -Pav {os.path.join(cur_dir, best_model_name)}model.h5 {root_dir}models/ ')
if cur_dir != '{root_dir}': 
    myexecute(f'rm -rf {cur_dir}')
