import argparse
from image_datagen.generator import *
from image_models.models.models import get_model
from image_models.losses.losses import ce_dice_loss, bce_dice_loss, dice_coeff, gen_dice_coeff
from keras.losses import binary_crossentropy, categorical_crossentropy

import tensorflow as tf
import keras
from keras.callbacks import *

from utils import *
import os
import json
import glob
import time

parser = argparse.ArgumentParser()

# model arguments
parser.add_argument('-m', '--model', help='Name of the model to train',
                    required=True)
parser.add_argument('-s', '--structure', help='Structure of the data', 
                    choices=['pair', 'stacked', 'sequence'])
parser.add_argument('-l', '--labeled', help='Use labeled data',
                    action='store_true', default=False)                    

# dataset arguments
parser.add_argument('-dt', '--dataset_type', help='Type of the dataset, used for creating a suitable data generator',
                    choices=['text', 'image', 'mixed'], default='text')
parser.add_argument('-dp', '--dataset_paths', nargs='+', help='Folders that contain data',
                    required=True)
parser.add_argument('-dh', '--dataset_input_height', help='Height dimension of input data to model',
                    type=int, default=256)
parser.add_argument('-dw', '--dataset_input_width', help='Width dimension of input data to model',
                    type=int, default=256)
parser.add_argument('-dn', '--dataset_input_channels', help='Channels dimension of input data to model',
                    type=int, default=1)
parser.add_argument('-dz', '--dataset_stack_size', help='Stack size of input data',
                    type=int, default=3)                    
parser.add_argument('-dc', '--dataset_num_classes', help='Number of classes for labels in dataset',
                    type=int, default=4)              
parser.add_argument('-da', '--dataset_crop_area', help='Area of the image covered by crops',
                    type=float, default=0.8)    
parser.add_argument('-dm', '--dataset_max_num', help='Maximum number of data files to use to limit dataset size',
                    type=int, default=None)  


# training arguments
parser.add_argument('-to', '--train_optimizer', help='Optimizer used for training',
                    default='adam')
parser.add_argument('-tr', '--train_learning_rate', help='Learning rate for optimizer',
                    type=float, default=0.001)                    
parser.add_argument('-tl', '--train_loss', help='Loss to train on',
                    default='crossentropy',
                    choices=['crossentropy', 'dice'])
parser.add_argument('-tc', '--train_crops', help='Use crops of the original data',
                    type=int, default=None)
parser.add_argument('-te', '--train_epochs', help='Number of epochs the model is trained for',
                    type=int, default=30)                                                            
parser.add_argument('-tb', '--train_batchsize', help='Batchsize used for training',
                    type=int, default=8)                    
parser.add_argument('-ts', '--train_split', help='Split between the training and validation data',
                    type=float, default=0.2)                    

args = parser.parse_args()

input_height = args.dataset_input_height
input_width = args.dataset_input_width
input_channels = args.dataset_input_channels
n_classes = args.dataset_num_classes
stacksize = args.dataset_stack_size

# Create and Configure model
if args.labeled:
    output_channels = n_classes
else:
    output_channels = input_channels

if output_channels > 1:
    if args.train_loss == 'crossentropy':
        loss = categorical_crossentropy
        metrics = ['accuracy']
    elif args.train_loss == 'dice':
        loss = ce_dice_loss
        metrics = [gen_dice_coeff, ]
    else:
        raise AttributeError("Unknown Loss")
else:
    if args.train_loss == 'crossentropy':
        loss = binary_crossentropy
        metrics = ['accuracy']
    elif args.train_loss == 'dice':
        loss = bce_dice_loss
        metrics = [dice_coeff, ]
    else:
        raise AttributeError("Invalid loss")

# Provide extra argument for stack size when network is recurrent
if args.model.startswith('lstm'):
    model, output_height, output_width = get_model(args.model, input_height, input_width, input_channels, output_channels, False, stacksize)
else:
    model, output_height, output_width = get_model(args.model, input_height, input_width, input_channels, output_channels)

if args.train_optimizer == 'adam':
    optimizer = keras.optimizers.Adam(lr=args.train_learning_rate)
elif args.train_optimizer == 'adadelta':
    optimizer = keras.optimizers.Adadelta(lr=args.train_learning_rate)
elif args.train_optimizer == 'sgd':
    optimizer = keras.optimizers.SGD(lr=args.train_learning_rate)
else:
    raise AttributeError("Invalid optimizer")

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model_output_dir = 'output/%s/%s/%s/%d' % (args.model, args.structure, "labeled" if args.labeled else "unlabeled", time.time())
if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir, exist_ok=True)

# save model as json
with open(model_output_dir + '/model.json', 'w') as f:
    f.write(model.to_json())

# save parameters
with open(model_output_dir + '/param.json', 'w') as f:
    d = dict()
    for key in vars(args):
        d[key] = getattr(args, key)
    f.write(json.dumps(d))

filepath = model_output_dir + '/weights_{epoch:03d}.hdf5'

print("Model output:", model_output_dir)

callbacks = []
callbacks.append(ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True))
callbacks.append(ReduceLROnPlateau(monitor='val_loss', patience=20, min_lr=0.001*args.train_learning_rate, 
                                   factor=0.1, verbose=1))
callbacks.append(EarlyStopping(monitor='val_loss', patience=100, verbose=1))
callbacks.append(TensorBoard(log_dir = model_output_dir, histogram_freq=0))

# Configure Keras
config = tf.ConfigProto()
config.gpu_options.allow_growth=True # [pylint: ignore]
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# calculate the crop scale 
orig_width = 1600
crop_scale = np.round((input_width/orig_width)/args.dataset_crop_area, 2)

# Create data generators
if args.dataset_type == 'text':
    gen_class = AM2018TxtGenerator
elif args.dataset_type == 'image':
    gen_class = AM2018ImageGenerator
elif args.dataset_type == 'mixed':
    gen_class = AM2018MixedGenerator
else:
    raise AttributeError("Generator type %s not supported" % args.dataset_type)

training_data = gen_class(args.dataset_paths, (input_height, input_width, input_channels * stacksize), 
                         (output_height, output_width, n_classes), crop_scale=crop_scale, 
                                      num_data=args.dataset_max_num)

# Convert 0 crops to None
if args.train_crops == 0:
    args.train_crops = None

tgen, vgen = training_data.generator(structure=args.structure, labeled=args.labeled, batch_size=args.train_batchsize, 
                                     num_crops=args.train_crops, split=args.train_split)

# Train model
print("Start Training on", len(training_data), "datapoints")
data = next(tgen)
print("Batches of ", data[0].shape, ",", data[1].shape)

t_steps = int((len(training_data) * (1 - args.train_split)) // args.train_batchsize)
v_steps = int((len(training_data) * args.train_split) // args.train_batchsize)

print("Training with", t_steps, "/", v_steps, "steps")
model.fit_generator(tgen,
                    epochs=args.train_epochs, 
                    steps_per_epoch=t_steps,
                    validation_data=vgen,
                    validation_steps=v_steps,                    
                    callbacks=callbacks,
                    shuffle=True, 
                    workers=2,
                    verbose=1)

print("Finish Training")

model.save_weights(model_output_dir + '/last_weights.hd5f')
best_weights = glob.glob(model_output_dir + '/weights*')[-1]

model.load_weights(best_weights)

print("Evaluate on Validation data")
score = model.evaluate_generator(vgen, steps=1, verbose=1)
print("Evaluation: Accuracy %.2f%%" % (score[1] * 100))