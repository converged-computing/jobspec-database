import os
import sys
import threading
from time import sleep

import tensorflow as tf

from Network import Network
from Worker import Worker
from env.Env import Env
from support.integrated_server import start_server

max_episode_length = 150
max_buffer_length = 30
gamma = .99  # discount rate for advantage estimation and reward discounting
height = 20
width = 20
depth = 8
s_size = height * width * depth  # Observations are greyscale frames of 84 * 84 * 1
a_size = len(Env.get_action_meanings())  # Agent can move in many directions
load_model = False
unsupervised = True
model_path = './model'
last_id_path = './last_ids'
num_workers = 20  # multiprocessing.cpu_count()  # Set workers ot number of available CPU threads
use_integrated_server = True

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Startup the server for fun and glory
if use_integrated_server:
    if not start_server():
        print('Kill process because server did not start')
        sys.exit(1)

global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, epsilon=0.1, decay=0.99)
master_network = Network(height, width, depth, s_size, a_size, 'global', None)  # Generate global network


print('Creating', num_workers, 'workers')
workers = []
# Create worker classes
for i in range(num_workers):

    # Only worker 0 are self_exploring
    workers.append(
        Worker(i, (height, width, depth, s_size), a_size, trainer, model_path, global_episodes, explore_self=unsupervised))
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()

    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver, max_buffer_length)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)
