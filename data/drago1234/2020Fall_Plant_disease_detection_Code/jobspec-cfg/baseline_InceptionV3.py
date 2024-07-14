# Note: This file contains three baseline model: VGG19, ResNet50, and InceptionV3. For more advance model, I suggest you to duplicate this file and add more there (Because this is the most stable version, being tested many times)
# Updated 12/18/2020 ==>  Note for data loading: Basically, there are two way to load the image dataset: 1) Using ImageDataGenerator,tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory() 2)  tf.keras.preprocessing.image_dataset_from_directory(). With first methods, you can dirrected scale the size with ImageDataGenerator. For second methods, it has similar utility to control the loading process, such as shuffling, batching, color_mode management. Two methods are similar in terms of the loading procedure, but the first one it's easier to apply the real-time image augmentation, with ImageDataGenerator. You still can apply the image augmentation with second methods with Module: tf.keras.layers.experimental.preprocessing, however that is required a little more work, so ==> It's more recommended to use the ImageDataGenerator. (Reference: Method1, https://www.tensorflow.org/hub/tutorials/tf2_image_retraining#select_the_tf2_savedmodel_module_to_use, Method2, https://www.tensorflow.org/tutorials/images/classification)



# Import package
import argparse
from collections import Counter
import logging
import math
import os
import random
import re
import shutil
from shutil import copyfile
import sys
import threading
import time
#import utils
import zipfile
# Data Science packages
import keras
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat


# Check the underlying hardware devices
print(f"Does the system is built with CUDA?: {tf.test.is_built_with_cuda()}")

# import tensorflow.compat.v1 as tf
import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


# Understand the file structure of dataset
DATA_PATH = '/fs/scratch/PAA0023/dong760/PlantVillage-Dataset/raw/color/'
# train_dir = "/fs/scratch/PAS1777/imdb_wiki/dataset/{}/train".format(db)
# test_dir = "/fs/scratch/PAS1777/imdb_wiki/dataset/{}/test".format(db)
Classes = os.listdir(DATA_PATH)
print(f"Classes: {Classes}")

# Understand the # of classes:
Num_Classes = len(Classes)
print(f"Number of classes: {Num_Classes}") # ==> Number of classes: 38

# Understand the total number of images
import pathlib
data_dir = pathlib.Path(DATA_PATH)
image_count = len(list(data_dir.glob('*/*.*')))
print(f"Total number of images: {image_count}")

# Display an images as sample:
# import PIL
# import PIL.Image
# roses = list(data_dir.glob('Corn_(maize)___Common_rust_/*'))
# PIL.Image.open(str(roses[0]))

# Let's figure out a way to understand the shape of image
folder_path = [os.path.join(DATA_PATH, path) for path in os.listdir(DATA_PATH)]
first_folder_path = folder_path[0]
print(f"first_folder_path: {first_folder_path}")
print(f"Number of folder: {len(folder_path)}")

img_path_list = [[os.path.join(folder, img_path) for img_path in os.listdir(folder)] for folder in folder_path]
first_img_path = img_path_list[0][0]
print(f"sample image path: {first_img_path}")

# Understand the shape of an image
img = mpimg.imread(first_img_path)
print(f"Image size: {img.shape}") # ==> Image size: (256, 256, 3)

# Display one image if you want to 
# plt.imshow(img, interpolation='nearest')
# plt.show()

# Get the total number of images with map reduce (refers to https://stackabuse.com/map-filter-and-reduce-in-python-with-examples/)
# from functools import reduce
# num_images = reduce(lambda x,arr2: (x+len(arr2)), img_path_list, 0)
# print(f"Total number of images: {num_images}")

# Visualize the dataset
# from matplotlib import pyplot as plt
# class_names = [[str(folder) for i in os.listdir(os.path.join(DATA_PATH, folder))] for folder in os.listdir(DATA_PATH)]
# images = [[os.path.join(folder, img_path) for img_path in os.listdir(folder)] for folder in folder_path]
# plt.figure(figsize=(10, 10))
# for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     img = mpimg.imread(images[0][i])
#     plt.imshow(img, interpolation='nearest')
#     plt.title(class_names[0][i])
#     plt.axis("off")
# plt.show()

# Model configuration
BASE_DIR = "/users/PAA0023/dong760/plant_leaves_diagnosis"
batch_size = 16
no_epochs = 5
img_width, img_height, img_num_channels = 256, 256, 3
loss_function = 'categorical_crossentropy' # 'mean_squared_error', 'mean_absolute_error', 'binary_crossentropy', 'sparse_categorical_crossentropy', 'categorical_crossentropy'
metrics = [tf.keras.metrics.Precision(), tf.keras.metrics.CategoricalAccuracy(name='accuracy')] # 'accuracy', 'precision', 'recall', 'auc', and more(https://www.tensorflow.org/tfx/model_analysis/metrics, https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile)
no_classes = len(os.listdir(DATA_PATH))
optimizer = tf.keras.optimizers.Adam(0.001) # SGD, Adagrad, RMSprop
validation_split = 0.2
verbosity = 2 # 0 = silent, 1 = progress bar, 2 = one line per epoch. 
steps_per_epoch = 100 # Total number of steps for one epochs
lr = 0.001
momentum=0.9

METRICS = [
#       tf.keras.metrics.TruePositives(name='tp'),
#       tf.keras.metrics.FalsePositives(name='fp'),
#       tf.keras.metrics.TrueNegatives(name='tn'),
#       tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc')
]
# Note: Read more aboue metrics configuration: 1) https://keras.io/guides/functional_api/#save-and-serialize, 2)https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics, 3) https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Precision, 4)https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile

DESIRED_ACCURACY = 0.99
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>DESIRED_ACCURACY):
            print("\nReached ",DESIRED_ACCURACY, "% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()


# Setting up the ImageGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory

train_datagen = ImageDataGenerator(rescale=1./255, 
                                   validation_split=validation_split,
                                   rotation_range=25,
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   fill_mode='nearest',
                                   horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)

train_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(img_width, img_height),  # (256, 256, 3)
    color_mode='rgb', #  "grayscale", "rgb", "rgba", Default: "rgb". Whether the images will be converted to have 1, 3, or 4 channels.
    class_mode = 'categorical', # One of "categorical", "binary", "sparse", "input", or None.
#     label_mode='int', # 'int': encoded as integer for sparse_categorical_crossentropy loss, 'categorical': encoded as categorical vector for categorical_crossentropy loss, 'binary': encoded as 0 or 1 for binary_crossentropy
    batch_size=batch_size,
    shuffle=True,
#     save_to_dir = '/users/PAA0023/dong760/plant_leaves_diagnosis/tmp/augmented_train', # This allows you to optionally specify a directory to which to save the augmented pictures being generated (useful for visualizing what you are doing).
    seed=123,
    subset="training",
    interpolation='nearest' #  Supported methods are "nearest", "bilinear", and "bicubic"
)
# print(type(train_generator))
# print(train_generator.shape)

validation_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(img_width, img_height),  # (256, 256, 3)
    color_mode='rgb', #  "grayscale", "rgb", "rgba", Default: "rgb". Whether the images will be converted to have 1, 3, or 4 channels.
    class_mode = 'categorical', # One of "categorical", "binary", "sparse", "input", or None.
#     label_mode='int', # 'int': encoded as integer for sparse_categorical_crossentropy loss, 'categorical': encoded as categorical vector for categorical_crossentropy loss, 'binary': encoded as 0 or 1 for binary_crossentropy
    batch_size=batch_size,
    shuffle=True,
#     save_to_dir = '/users/PAA0023/dong760/plant_leaves_diagnosis/tmp/augmented_valid', # This allows you to optionally specify a directory to which to save the augmented pictures being generated (useful for visualizing what you are doing).
    seed=123,
    subset="validation",
    interpolation='nearest' #  Supported methods are "nearest", "bilinear", and "bicubic"
)
# print(type(validation_generator))
# print(validation_generator.shape)

# Define some variable for Horovod
train_iterator = train_generator
train_size = train_iterator.n # OR len(train_iterator.filepaths), len(train_iterator.classes), len(train_iterator.filenames)
# batch_size = train_iterator.batch_size
# val_size = validation_iterator.n
# len(train_iterator.filepaths)
# len(train_iterator.classes)
# train_iterator.num_classes
# train_iterator.image_shape
# train_iterator.batch_size
# train_iterator.dtype
print(f"Train size: {train_size}")


# Configure the pre-trained models:
MODEL_NAME = 'InceptionV3' # ResNet, NASNet, InceptionV3
IMG_SHAPE = (img_width, img_height, img_num_channels)
# For vgg19
VGG19_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
# For ResNet50
ResNet50_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
# For InceptionV3
InceptionV3_model = InceptionV3(input_shape=IMG_SHAPE,
                                 include_top = False,
                                 weights = 'imagenet') 

# Lock each layer in pre_trained_model
for layer in InceptionV3_model.layers:
    layer.trainable = False

# If you want to do fine-tunnning
# For VGG19
# VGG19_last_layer = VGG19_model.get_layer('block5_pool')
# For InceptionV3
# pre_trained_model.load_weights(local_weights_file) # Load the weights from previously downloaded file
# InceptionV3_mixed7_layer = InceptionV3_model.get_layer('mixed7')

# Fine tuning the model: Define the new model by adding extra fully connected layers
last_output = InceptionV3_model.output
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
output_layer = tf.keras.layers.Dense(no_classes, activation='softmax')(x)
model = tf.keras.Model(inputs=InceptionV3_model.input, outputs=output_layer)


# Instantiate the Model object: (More detail about compile: https://www.tensorflow.org/guide/keras/train_and_evaluate)
# optimizer = tf.keras.optimizers.SGD(lr=lr, momentum=momentum)
# optimizer = tf.keras.optimizers.RMSprop(lr=0.0001)
optimizer =  tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer,
              loss=loss_function,
              metrics=METRICS)
print(f"\n====> Statistics: MODEL_NAME={MODEL_NAME}, epochs={no_epochs}, batch_size={batch_size}, validation_split={validation_split}, lr={lr}, momentum={momentum}, steps_per_epoch ={steps_per_epoch}, feature shape= {(img_width, img_height, img_num_channels)}, no_classes={no_classes }, loss_function={loss_function}")
# print(model.summary())

# Define the checkpoint directory:
checkpoint_dir = './checkpoints/'
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

# Training the model:
history = model.fit(train_generator,
                    epochs=no_epochs, 
                    verbose=verbosity,
#                     callbacks=[callbacks],
#                     steps_per_epoch=steps_per_epoch,                    
#                     validation_steps=steps_per_epoch,
                    validation_data=validation_generator)
print(history.history)

# Evaluate the performance
# print("\n====> Evaluate the Validation performance: ")
# valid_loss, valid_acc = model.evaluate(validation_generator, verbose=verbosity)
# print('Validation accuracy:', valid_acc, "Validation accuracy:", valid_loss)

# Get the timestamp
from datetime import datetime 
now = datetime.now()
# print("now =", now)
# dd/mm/YY H:M:S
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
print("date and time =", dt_string)	
# dd/mm/YY
# today = date.today()
# d1 = today.strftime("%d/%m/%Y")
# print("d1 =", d1)

# Plot the figure
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='best')
plt.ylabel('Accuracy')
# plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='best')
plt.ylabel('Loss')
# plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
# plt.show()
plt.savefig(BASE_DIR+"/plots/"+MODEL_NAME+"_BatchSize_"+str(batch_size)+"_"+str(validation_split)+"ValSplit_"+str(dt_string)+"_"+str(no_epochs)+"epochs"+'.png')

# def plot_graphs(history, string, filename):
#     plt.plot(history.history[string])
#     plt.plot(history.history['val_'+string])
#     plt.xlabel("Epochs")
#     plt.ylabel(string)
#     plt.legend([string, 'val_'+string])
#     plt.savefig('/users/PAA0023/dong760/plant_leaves_diagnosis/plots/'+MODEL_NAME+"_batchsize_"+str(batch_size)+"_"+str(dt_string)+'.png')
#     plt.show()
# plot_graphs(history, 'acc', 'test01')
# plot_graphs(history, 'loss', 'test01')

# Saving the model includes the: - model architecture - model weight values (that were learned during training) - model training config, if any (as passed to compile) - optimizer and its state, if any (to restart training where you left off)
saved_model_path = BASE_DIR+"/saved_models/"+MODEL_NAME+"_BatchSize_"+str(batch_size)+"_"+str(validation_split)+"ValSplit_"+str(dt_string)+"_"+str(no_epochs)+"epochs"
tf.saved_model.save(model, saved_model_path)

# Saving the model task graphs
dot_img_file = BASE_DIR+"/model_graphs/"+MODEL_NAME+"_BatchSize_"+str(batch_size)+"_"+str(validation_split)+"ValSplit_"+str(dt_string)+"_"+str(no_epochs)+"epochs"+'.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

print(f"Saving the model to path: {saved_model_path}")

# =================> Do the unimportant things at last
# Evaluate the performance
print("\nPrediction Result: ")
result = model.evaluate(validation_generator, verbose=verbosity)
print(f"Result: {result}")
print('Validation accuracy:', result[1], "Validation accuracy:", result[0])