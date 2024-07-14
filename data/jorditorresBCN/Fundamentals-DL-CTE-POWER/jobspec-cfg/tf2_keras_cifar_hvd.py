import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import horovod.tensorflow.keras as hvd
import numpy as np
import argparse
import time
import sys

sys.path.append('/gpfs/projects/nct00/nct00010/cifar-utils')
from cifar import load_cifar

hvd.init()

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--model_name', type=str, default='resnet')

args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs
model_name = args.model_name

print(' model:', model_name)
print(' batch_size:', batch_size)
print(' epochs:', epochs)

#model_names = ['resnet', 'vgg']
#assert model_name in model_names, f'model_name must to be one of these: {model_names}'

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

train_ds, test_ds = load_cifar(batch_size)


if model_name == 'resnet':
    model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=True, weights=None,
            input_shape=(128, 128, 3), classes=10)
elif model_name == 'vgg':
    model = tf.keras.applications.VGG16(include_top=True, weights=None,
            input_shape=(128, 128, 3), classes=10)
            
#if hvd.rank() == 0:
#    print(model.summary())

opt = tf.keras.optimizers.SGD(0.0005 * hvd.size())
opt = hvd.DistributedOptimizer(opt)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
    experimental_run_tf_function=False)
    
callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0)
]

if hvd.rank() == 0:
    verbose = 2
else:
    verbose=0

start = time.time()   
model.fit(train_ds, epochs=epochs, verbose=verbose, callbacks=callbacks)
end = time.time()
if hvd.rank() == 0:
    print('Total Time:', round((end - start), 2), '(s)')
