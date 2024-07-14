#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import numpy as np
from pathlib import Path
from datetime import datetime
import os


DATA_PATH = Path("/scratch/ajk4yq/ecg/tfrecords/")
TRAIN_RECS = list(DATA_PATH.glob("train*.tfrecords"))
VAL_RECS = list(DATA_PATH.glob("val*.tfrecords"))


BATCH_SIZE = 8

record_format = {
    'ecg/data': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'age': tf.io.FixedLenFeature([], tf.float32),
    'gender': tf.io.FixedLenFeature([], tf.int64),
}

def _parse_record(record):
    example = tf.io.parse_single_example(record, record_format)
    ecg_data = tf.reshape(example['ecg/data'], [5000,12])
    label = example['age']
    return ecg_data, label

def drop_na_ages(x,y):
    return not tf.math.reduce_any(tf.math.is_nan(y))

def age_lt_90(x,y):
    return tf.math.reduce_all(tf.math.less_equal(y, tf.constant([90.0])))

def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(_parse_record, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(drop_na_ages)
    dataset = dataset.filter(age_lt_90)
    return dataset

def get_dataset(filenames, labeled=True):
    dataset = load_dataset(filenames)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


# In[ ]:


train_dataset = get_dataset(TRAIN_RECS)
val_dataset = get_dataset(VAL_RECS)


# In[ ]:


#https://keras.io/examples/timeseries/timeseries_transformer_classification/
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads,dropout=dropout)(x,x)
    x = tf.keras.layers.Dropout(dropout)(x)
    res = x + inputs
    
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res
    
input_layer = tf.keras.layers.Input(shape=(5000, 12))
x = input_layer
x = transformer_encoder(x, head_size=5000, num_heads=2, ff_dim=4*12, dropout=0)
x = tf.keras.layers.MaxPooling1D()(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(64)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1)(x)

model = tf.keras.models.Model(input_layer, x)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['mse', 'mae']
)


# In[ ]:


model.summary()


# In[ ]:


def make_checkpoint_dir(data_path, label):
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"{label}-{formatted_datetime}"
    output_path = f"{data_path}/{output_dir}"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    return output_path


callbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.ReduceLROnPlateau(),
    tf.keras.callbacks.ModelCheckpoint(make_checkpoint_dir("data/models", "transformer-age3"))
]

model.fit(train_dataset, epochs=15, validation_data=val_dataset, callbacks=callbacks)