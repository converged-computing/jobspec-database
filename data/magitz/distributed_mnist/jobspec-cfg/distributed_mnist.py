#!/usr/bin/env python
# coding: utf-8

# # Distributed training with Keras example using MNIST dataset
# 
# This tutorial is adapted from the TensorFlow tutorial here: [https://www.tensorflow.org/tutorials/distribute/keras](https://www.tensorflow.org/tutorials/distribute/keras)
# 
# To run this on HiPerGator, I requested an [Open onDemand](https://ood.rc.ufl.edu/) Jupyter session with:
# * 4 cores
# * 64 GB RAM
# * 2 A100 GPUs: `gpus:a100:2`
# 
# This tutorial is intended to help get you up and running training models on HiPerGator and to show how simple it can be to scale things up as you exceed the capabilities of a single GPU. This tutorial also introduced [TensorBoard](https://www.tensorflow.org/tensorboard/get_started#:~:text=TensorBoard%20is%20a%20tool%20for,dimensional%20space%2C%20and%20much%20more.), a relatively easy method of tracking model training and your hyperparameter tuning experiments.
# 
# ## Load some modules and check TensorFlow version

# In[2]:


import tensorflow_datasets as tfds
import tensorflow as tf

import os

# Load the TensorBoard notebook extension.
# get_ipython().run_line_magic('load_ext', 'tensorboard')  #### Need to comment this out for batch runs!


# In[3]:


print(tf.__version__)


# ## Load the MNIST dataset
# 
# Remember the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) is the handwritten image dataset that was featured in the NVIDIA Foundations of Deep Learning (or Getting Started with Deep Learning) course.
# 
# ![Sample images from the MNIST dataset, Image by 
# Suvanjanprasai from Wikipedia](images/MnistExamplesModified.png)
# 
# We don't really need multiple GPUs to analyze these data, but they are handy to use and a familiar example.

# In[4]:


datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = datasets['train'], datasets['test']


# ## For multi-GPUs, setup a strategy
# 
# There are many different ways of using multiple GPUs in training a model. As with many things, without a framework, things can get complex quickly. Luckily Keras provides a nice API framework to do most of the work for us.
# 
# First, there are two main *things* that can be distributed across multiple GPUs:
# * **Data parallelism** uses multiple GPUs to train a single model with each GPU evaluating different batches of the dataset and averaging results periodically.
# * **Model parallelism** where a model is split across multiple GPUs. Models can be split with different layers on different GPUs, splitting the weights of a single layer across GPUs, or both.
# 
# As you scale beyond a single GPU, it is easier (and more efficient) to use multiple GPUs in a single server (8 in the DGX Servers), but after that, you need to use GPUs on multiple servers (hosts).
# 
# See more details [here](https://www.tensorflow.org/guide/keras/distributed_training).
# 
# The cool thing is, that setting up multi-GPU training, can be quite simple. In this case, we really only need a couple of lines:

# In[5]:


# The MirroredStrategy works for multiple GPUs on 1 server.
strategy = tf.distribute.MirroredStrategy()


# In[6]:


# Double check how many GPUs our strategy sees
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


# ## Set up the input pipeline
# 
# As noted in the [original tutorial](https://www.tensorflow.org/tutorials/distribute/keras#set_up_the_input_pipeline), we can increase the batch size:
# > When training a model with multiple GPUs, you can use the extra computing power effectively by increasing the batch size. In general, use the largest batch size that fits the GPU memory and tune the learning rate accordingly.
# 
# My guess is that on the A100s, we can go larger than 64, but let's start there.

# In[7]:


# You can also do info.splits.total_num_examples to get the total
# number of examples in the dataset.

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


# In[8]:


def scale(image, label):
   '''Define a function that normalizes the image pixel values from the [0, 255] range to the [0, 1] range'''
   image = tf.cast(image, tf.float32)
   image /= 255

   return image, label


# In[9]:


#Apply this scale function to the training and test data, 
# and then use the tf.data.Dataset APIs to shuffle the training 
# data (Dataset.shuffle), and batch it (Dataset.batch). 
# Notice that you are also keeping an in-memory cache of the training 
# data to improve performance (Dataset.cache).

train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)


# ## Create and compile the model
# 
# The main change here is that we make the model within the context of the `Strategy.scope`, but making and compiling the model is the same as it would be on a single GPU.

# In[11]:


with strategy.scope():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
  ])

  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'])


# Another interesting note in the tutorial about learning rate and using multiple GPUs:
# > For this toy example with the MNIST dataset, you will be using the Adam optimizer's default learning rate of 0.001.
# >
# > For larger datasets, the key benefit of distributed training is to learn more in each training step, because each step processes more training data in parallel, which allows for a larger learning rate (within the limits of the model and dataset).

# ## Callbacks
# 
# Callbacks are the tools we can use to monitor training, control when to stop and lots of other useful features. We'll use some handy ones here.

# In[15]:


# Define the checkpoint directory to store the checkpoints.
# This setus up logging for TensorBoard
checkpoint_dir = './training_checkpoints'
# Define the name of the checkpoint files.
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")


# In[16]:


# Define a function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7: # Drop the learning rate for epochs 3-6
    return 1e-4
  else:                         # Drop the learning rate again for epochs 7 and on
    return 1e-5


# In[17]:


# Define a callback for printing the learning rate at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(        epoch + 1, model.optimizer.lr.numpy()))


# In[18]:


# Put all the callbacks together.
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]


# In[21]:


EPOCHS = 12

model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks)


# In[22]:


# Check the checkpoint directory.
# get_ipython().system('ls {checkpoint_dir}')  #### Need to comment this out for batch runs!


# In[23]:


model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

eval_loss, eval_acc = model.evaluate(eval_dataset)

print('Eval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))


# ## Look at training in TensorBoard
# 
# Because of the way we are connected to the session with Jupyter, we can't easily use this session for running TensorBoard. Let's [follow the directions here to open a new HiPerGator Desktop](https://help.rc.ufl.edu/doc/Tensorboard) to look at the TensorBoard logs.
# 
# * In a new tab, go to [https://ood.rc.ufl.edu/](https://ood.rc.ufl.edu/)
# * From the Interactive Apps menu, select the 1st option: HiPerGator Desktop
# * Request:
#    * 2 cores
#    * 12 GB RAM
#    * 1 hour of time
# * Click Launch
# * When the job starts, click the **Launch HiPerGator Desktop** button
# * That will open a GUI Desktop in a new session.
# * From the Applications menu, select **Terminal Emulator**
# * Change directories to the directory where this notebook is located. e.g. `cd /blue/gms6029/$USER/distributed_mnist`
# * Load the needed modules: `module load tensorflow ubuntu`
# * Launch TensorBoard: `tensorboard --logdir=./logs &` 
#     * The `&` tells bash to return rather than wait for the command to complete
# * Wait a few seconds and once you see a line that says `TensorBoard 2.7.0 at http://localhost:6006/ (Press CTRL+C to quit)`, hit return to get the command prompt back
# * Type: `firefox` to open the browser
# * Go to `http://localhost:6006/`
# 
# 

# In[ ]:




