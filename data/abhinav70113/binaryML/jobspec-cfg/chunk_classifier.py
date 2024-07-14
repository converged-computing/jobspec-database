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
import json
from tensorflow.keras.layers import MultiHeadAttention
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import Callback
from BaseClass import Models
import pandas as pd
def myexecute(cmd):
    if 'echo' not in cmd[:20]:
        os.system("echo '%s'"%cmd)
    os.system(cmd)


job_id = sys.argv[1]
model_type = sys.argv[2]
run = 'runBD'
index = 0
cur_dir = f'/tmp/Abhinav_DATA{job_id}/'
#model_type = 'LSTM'
input_shape = (800,1)
batch_size = 400
epochs = 20000
patience = 100
snr_group = 1
best_model_name = f'{cur_dir}models/chunk_classify_{model_type}_{job_id}_'
root_dir = '/hercules/scratch/atya/BinaryML/'

list_of_dicts = json.load(open(f'{root_dir}hyperparameter_tuning/cnn/list_of_dicts.json'))
param_dict = list_of_dicts[index]

myexecute(f'echo \"\n\n\n\n############################################################################## \n\n\n \
          Comment: Binary simulations predictor: Classifying chunks \nBest model is under:{best_model_name} \
           \n\n\n ############################################################################## \n\n\n \"')

myexecute(f'mkdir -p {cur_dir}raw_data/{run}/')
myexecute(f'mkdir -p {cur_dir}models/')

#files to sync
files1 = glob.glob(f'{root_dir}raw_data/{run}/*classifier_{input_shape[0]}.npy')
#files1.extend(glob.glob(f'{root_dir}raw_data/{run}/*indices.npy'))
for file in files1:
    myexecute(f'rsync -Pav -q {file} {cur_dir}raw_data/{run}/')

# freq_axis = np.fft.rfftfreq(17280000, d=64e-6)
# freq_res = freq_axis[1]-freq_axis[0]

X_train = np.load(cur_dir + f'raw_data/{run}/train_data_classifier_{input_shape[0]}.npy').astype(np.float64)
X_test = np.load(cur_dir + f'raw_data/{run}/test_data_classifier_{input_shape[0]}.npy').astype(np.float64)
X_val = np.load(cur_dir + f'raw_data/{run}/val_data_classifier_{input_shape[0]}.npy').astype(np.float64)
X_train = X_train/np.max(X_train,axis=1)[:,None]
X_test = X_test/np.max(X_test,axis=1)[:,None]
X_val = X_val/np.max(X_val,axis=1)[:,None]

Y_train = np.load(cur_dir + f'raw_data/{run}/train_labels_classifier_{input_shape[0]}.npy').astype(np.float64)
Y_test = np.load(cur_dir + f'raw_data/{run}/test_labels_classifier_{input_shape[0]}.npy').astype(np.float64)
Y_val = np.load(cur_dir + f'raw_data/{run}/val_labels_classifier_{input_shape[0]}.npy').astype(np.float64)


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1) 

Y_train = (Y_train.reshape(Y_train.shape[0],1))
Y_val = (Y_val.reshape(Y_val.shape[0],1))
Y_test = (Y_test.reshape(Y_test.shape[0],1))


train_indices = np.load(cur_dir + f'raw_data/{run}/train_indices_classifier_{input_shape[0]}.npy')
test_indices = np.load(cur_dir + f'raw_data/{run}/test_indices_classifier_{input_shape[0]}.npy')
val_indices = np.load(cur_dir + f'raw_data/{run}/val_indices_classifier_{input_shape[0]}.npy')

def find_indices(small_list, big_list):
    indices = []

    for item in small_list:
        if item in big_list:
            index_list = np.where(big_list == item)
            indices.extend(index_list[0])

    return indices

if snr_group != -1:

    labels_df = pd.read_csv(root_dir + f'meta_data/labels_{run}.csv')
    snr_range = labels_df['fold_snr_theory'].values
    #snr_bins = np.array([0 , 0.068, 0.126, 0.184, 0.242, 0.3  ])
    snr_bins = np.array([0 , 5, 30])

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

# find 0 and 1 indices in Y_train
Y_train_0 = np.where(Y_train == 0)[0]
Y_train_1 = np.where(Y_train == 1)[0]
Y_train_01 = np.concatenate((Y_train_0, Y_train_1), axis=0)
np.random.shuffle(Y_train_01)
Y_train = Y_train[Y_train_01]
X_train = X_train[Y_train_01]

# find 0 and 1 indices in Y_val
Y_val_0 = np.where(Y_val == 0)[0]
Y_val_1 = np.where(Y_val == 1)[0]
Y_val_01 = np.concatenate((Y_val_0, Y_val_1), axis=0)
np.random.shuffle(Y_val_01)
Y_val = Y_val[Y_val_01]
X_val = X_val[Y_val_01]

# find 0 and 1 indices in Y_test
Y_test_0 = np.where(Y_test == 0)[0]
Y_test_1 = np.where(Y_test == 1)[0]
Y_test_01 = np.concatenate((Y_test_0, Y_test_1), axis=0)
np.random.shuffle(Y_test_01)
Y_test = Y_test[Y_test_01]
X_test = X_test[Y_test_01]
myexecute(f'echo "Training data shape: {X_train.shape}"')
myexecute(f'echo "Test data shape: {X_test.shape}"')
myexecute(f'echo "Validation data shape: {X_val.shape}"')
myexecute(f'echo "Training labels shape: {Y_train.shape}"')
myexecute(f'echo "Test labels shape: {Y_test.shape}"')
myexecute(f'echo "Validation labels shape: {Y_val.shape}"')
print(Y_test[0:10])


# Create data generators
def create_dataset(X, y, batch_size, shuffle):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

train_generator = create_dataset(X_train, Y_train, batch_size=batch_size, shuffle=True)
val_generator = create_dataset(X_val, Y_val, batch_size=batch_size, shuffle=False)

def custom_tanh_activation(x):
    scaled_tanh = (tf.math.tanh(x) + 1) / 2
    range_scale = 0.029 / tf.reduce_max(scaled_tanh)  # maximum value is scaled to 0.029
    return range_scale * scaled_tanh + 0.001  # output is shifted by 0.001

def custom_sigmoid_activation(x):
    return 30 / (1 + K.exp(-x)) + 0.01

def weighted_mse_loss(y_true, y_pred):
    # Calculate squared error
    y_true = y_true
    y_pred = y_pred
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    squared_error = mse(y_true, y_pred)
    
    # Calculate the weighting factor (you can adjust the scaling factor as needed)
    weighting_factor = 1.0 / (y_true + 1e-8)  # Adding a small constant to avoid division by zero
    
    # Apply the weighting factor to the squared error
    weighted_squared_error = weighting_factor * squared_error
    
    # Calculate the mean of the weighted squared error
    weighted_mse = tf.reduce_mean(weighted_squared_error)
    
    return weighted_mse

def period_accuracy(y_true, y_pred):
    # Ensure the predictions are the same shape as the true values
    assert y_true.shape == y_pred.shape
    return accuracy_score(y_true, y_pred)

def custom_tanh_activation(x):
    scaled_tanh = (tf.math.tanh(x) + 1) / 2
    range_scale = 30 / tf.reduce_max(scaled_tanh)  # maximum value is scaled to 0.029
    return range_scale * scaled_tanh + 0.00001  # output is shifted by 0.001

def custom_sigmoid_activation(x):
    return 1 / (1 + K.exp(-x)) + 0.00001

def custom_relu_activation(x):
    return K.minimum(K.maximum(0.00001, x), 30)

# Define a helper function for a Residual block
def ResidualBlock(x, filters, kernel_size):
    skip = x
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    skip = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding='same')(skip) # to match dimensions
    return tf.keras.layers.Add()([x, skip]) # Skip connection
def LSTM_model(input_shape = (400,2)):

    # Model architecture
    inputs = tf.keras.Input(shape=input_shape)
    x = ResidualBlock(inputs, filters=64, kernel_size=5)
    x = ResidualBlock(x, filters=256, kernel_size=7)

    # Convert the output shape of ResidualBlock to 3D for LSTM
    x = tf.keras.layers.Reshape((-1, x.shape[-1]))(x)

    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(192, return_sequences=False)(x)
    outputs = tf.keras.layers.Dense(2, activation='relu')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def attention_model(input_shape = (400,2)):
    # Model architecture
    inputs = tf.keras.Input(shape=input_shape)

    # Apply Conv1D layers
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same',activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same',activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, padding='same',activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=7, padding='same',activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=7, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=512, kernel_size=7, padding='same',activation='relu')(x)
    # Apply Multihead Attention
    x = MultiHeadAttention(num_heads=2, key_dim=256)(x,x)
    #x = MultiHeadAttention(num_heads=4, key_dim=2)(x,x)
    #x = MultiHeadAttention(num_heads=8, key_dim=2)(x,x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(192, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def dense_model(input_shape = (400,2)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)

    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)  # Add this Flatten layer

    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)  # change activation function to 'sigmoid'
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def cnn_model(input_shape = (400,2)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same',dilation_rate = 2,activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=7, padding='same',dilation_rate = 4,activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=7, padding='same',dilation_rate = 8, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=512, kernel_size=7, padding='same',dilation_rate = 16, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=512, kernel_size=7, padding='same',dilation_rate = 32, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=512, kernel_size=7, padding='same',dilation_rate = 64, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=512, kernel_size=7, padding='same',dilation_rate = 64, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=512, kernel_size=7, padding='same',dilation_rate = 64, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def resnet_model(input_shape = (400,2)):
    inputs = tf.keras.Input(shape=input_shape)
    x = ResidualBlock(inputs, filters=64, kernel_size=3)
    x = ResidualBlock(x, filters=256, kernel_size=3)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)  # Add this Flatten layer

    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    #x = tf.keras.layers.Dense(1, activation='relu')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

param_dict['input_shape'] = input_shape
models = Models(param_dict)
def current_model(model_name = "deep",input_shape = (400,2)):
    if model_name == 'attention':
        return attention_model(input_shape=input_shape)
    elif model_name == 'deep':
        return dense_model(input_shape=input_shape)
    elif model_name == 'resnet':
        return resnet_model(input_shape=input_shape)
    elif model_name == 'cnn':
        #return models.cnn()
        return cnn_model(input_shape=input_shape)
    elif model_name == 'LSTM':
        return LSTM_model(input_shape=input_shape)
    else:
        raise ValueError('Model name not found')

model = current_model(model_name = model_type,input_shape=input_shape)
######################################

#Learning rate exponentia

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000,
    decay_rate=0.5)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# Set up the ModelCheckpoint callback
checkpoint_filepath = best_model_name+'checkpoint.h5'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Define early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, verbose=1, mode='max', restore_best_weights=True)

class MaxAccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') == 1.0:  # or 'val_accuracy' if you want to monitor the validation accuracy
            print("\nReached 100% accuracy, stopping training!")
            self.model.stop_training = True

max_accuracy_callback = MaxAccuracyCallback()

# Train the model
model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=[early_stop, checkpoint_callback, max_accuracy_callback], verbose=2)

model.save(best_model_name+'model.h5')

# Load the best model after training
#best_model = tf.keras.models.load_model(best_model_name+'checkpoint.h5', custom_objects={'custom_tanh_activation': custom_tanh_activation,'weighted_mse_loss': weighted_mse_loss, 'custom_sigmoid_activation': custom_sigmoid_activation,
#                                                                                         'weighted_mse_loss_wrapper' : get_weighted_mse_loss(default_hyperparameters['power'],default_hyperparameters['factor_mse'])})

best_model = model
Y_pred = (best_model.predict(X_test))
Y_pred = np.argmax(Y_pred, axis=-1)
Y_test = Y_test.reshape(Y_test.shape[0],)

accuracy = period_accuracy(Y_test, Y_pred)

myexecute(f'echo \"\n\n\n\n############################################################################## \n\n\n \
          The accuracy of the model is:{accuracy} \n\
          \n\n\n ############################################################################## \n\n\n \"')

myexecute(f'rsync -q -Pav {os.path.join(cur_dir, best_model_name)}checkpoint.h5 {root_dir}models/ ')
myexecute(f'rsync -q -Pav {os.path.join(cur_dir, best_model_name)}model.h5 {root_dir}models/ ')
if cur_dir != '{root_dir}': 
    myexecute(f'rm -rf {cur_dir}')
