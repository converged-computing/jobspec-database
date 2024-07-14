import sys
import os
import numpy as np
import tensorflow as tf
from autoencoder.EncoderData import DataLoader, batch_to_lists
from env.Env import Env
from autoencoder.EncoderNetwork import EncoderNetwork

episode_max = 1000000
skip_train_sets = 0
batch_size = 64
gamma = .99  # discount rate for advantage estimation and reward discounting
height = 20
width = 20
depth = 8
s_size = height * width * depth  # Observations are greyscale frames of 84 * 84 * 1
a_size = len(Env.get_action_meanings())  # Agent can move in many directions
r_size = len(Env.get_reward_meanings())  # number of different types of rewards we can get
# 0 = -1
# 1 = -0.1
# 2 = 1
# 3 = 10
load_model = False
model_path = './enc_model'
batch_path = '../batches/'
metadata_file = 'data.csv'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

trainer = tf.train.RMSPropOptimizer(learning_rate=7e-4, epsilon=0.1, decay=0.99)
network = EncoderNetwork(height, width, depth, s_size, a_size, r_size, batch_size, 'global', trainer)
increment = network.episodes.assign_add(1)

saver = tf.train.Saver(max_to_keep=5)
data = DataLoader(metadata_file, batch_size, batch_path, skip_train_sets)

with tf.Session() as sess:
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # Train train train
    # Chuu chuu
    summary_writer = tf.summary.FileWriter("Autoencoder")

    episode_enc_loss = []
    episode_val_loss = []
    episode = sess.run(network.episodes)

    while episode < episode_max:
        # increment episodes, storing episode count as a 'weight'
        sess.run(increment)
        episode += 1

        # get data
        batch = data.get_train()

        if batch is None:
            data = DataLoader(metadata_file, batch_size, batch_path, skip_train_sets)
            print('Epoch completed, restarting training at episode {}'.format(episode))
            saver.save(sess, model_path + '/model-' + str(episode) + '.cptk')
            continue

        x_state, x_action, y_state, y_reward = batch_to_lists(batch, s_size)

        enc_loss, val_loss = network.train(x_state, x_action, y_state, y_reward, sess)
        episode_enc_loss.append(enc_loss)
        episode_val_loss.append(val_loss)

        if episode % 5 == 0:

            batch = data.get_test()
            x_state, x_action, y_state, y_reward = batch_to_lists(batch, s_size)
            test_enc_loss, test_val_loss, test_rounded_loss = network.test(x_state, x_action, y_state, y_reward, sess)

            mean_enc_loss = np.mean(episode_enc_loss[-5:])
            mean_val_loss = np.mean(episode_val_loss[-5:])
            summary = tf.Summary()
            summary.value.add(tag="Encoding Loss (train)", simple_value=float(mean_enc_loss))
            summary.value.add(tag="Reward Loss (train)", simple_value=float(mean_val_loss))
            summary.value.add(tag="Encoding Loss (test)", simple_value=float(test_enc_loss))
            summary.value.add(tag="Reward Loss (test)", simple_value=float(test_val_loss))
            summary.value.add(tag="Encoding rounded (test)", simple_value=float(test_rounded_loss))
            summary_writer.add_summary(summary, episode)
            summary_writer.flush()

            if episode % 100 == 0:
                print(
                    'Episodes: {}, Encoding loss: {}, Encoding rounded loss: {}, Value loss: {}, Test encoding loss: {}, Test value loss: {}'
                    .format(episode, episode_enc_loss[-1], test_rounded_loss, episode_val_loss[-1], test_enc_loss,
                            test_val_loss))
                sys.stdout.flush()

    print('Episodes: {} TERMINATE'.format(episode))
