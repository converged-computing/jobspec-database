import os
import sys

if len(sys.argv) > 2:
    if sys.argv[2] == 'tf':
        os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, LSTM, Flatten, BatchNormalization
from keras.layers import Embedding
from keras.engine import Input
from keras.optimizers import RMSprop, SGD
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l1l2, activity_l1l2
import keras.models

from utils import *
from datetime import datetime

import ple
import ple.games.flappybird
import pygame.display

import io
import time
import random

if sys.version_info[0] < 3:
    import cPickle
else:
    import _pickle as cPickle


const_is_playback = False
const_is_load_model = False

const_frames_back = 5 #30

const_epsilon_explore_start = 0.2
const_epsilon_explore_end = 0.001
const_epsilon_explore_decay_frames = 3000000
const_epsilon_explore_decay_step = (const_epsilon_explore_start - const_epsilon_explore_end) / float(const_epsilon_explore_decay_frames)

const_memmory_buffer_max = 50000
const_startup_frames = 10000
const_frames_max = 50000

const_mini_batch_size = 32
const_batch_size = 32

const_rl_gamma = 0.99
const_is_frame_diff = False

const_lr = 1e-6
const_l1l2_regularization = 1e-2


const_nn_type = 'relu'
const_version = 'v7-online'

if len(sys.argv) > 1:
    if sys.argv[1] == 'lstm':
        const_nn_type = 'lstm'
    if sys.argv[1] == 'diff':
        const_frames_back = 1
        const_version += '-diff'
        const_is_frame_diff = True

epsilon_explore = 0

const_version += "-" + const_nn_type + "-" + str(const_lr) + "-" + str(const_l1l2_regularization) + "-" + str(const_frames_back)

init_log('./logs/rl-{0}-{1}.log'.format(const_version, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
file_csv_loss = open('./results/loss-{}-{}.csv'.format(const_version, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), 'a')

os.environ['SDL_AUDIODRIVER'] = "waveout"

if not const_is_playback:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.display.init()
    screen = pygame.display.set_mode((1,1))

def process_state(state):
    return np.array( list(state.values()) ).astype(np.float32)

def get_action(q_values):
    global actions_available, dimensions_actions

    if np.random.random() < epsilon_explore:
        if const_is_playback:
            logging.info('random action')

        rand_index = np.random.randint(0, dimensions_actions)
        q_values = np.zeros((dimensions_actions,)).tolist()
        q_values[rand_index] = 1.0
        return actions_available[rand_index], q_values

    if const_is_playback:
        logging.info('{} {}'.format(q_values[0], q_values[1]))

    return actions_available[np.argmax(q_values)], q_values


game = ple.games.flappybird.FlappyBird()
env = ple.PLE(game, display_screen=const_is_playback, state_preprocessor=process_state)
env.init()

dimensions_state = env.getGameStateDims()[0] + 1 #prev action
actions_available = env.getActionSet()
dimensions_actions = len(actions_available)

logging.info('model build started')

model = Sequential()

if const_is_load_model and os.path.exists('model-{}.h5'.format(const_version)):
    model = keras.models.load_model('model-{}.h5'.format(const_version))
    logging.info('model loaded')
else:
    if const_nn_type == 'lstm':
        model.add(LSTM(output_dim=100, input_dim=dimensions_state, input_length=const_frames_back, return_sequences=True,
                       W_regularizer=l1l2(const_l1l2_regularization), U_regularizer=l1l2(const_l1l2_regularization)))
        model.add(LSTM(output_dim=100, W_regularizer=l1l2(const_l1l2_regularization),
                       U_regularizer=l1l2(const_l1l2_regularization)))
    else:
        model.add(TimeDistributed(Dense(500, activation='relu', W_regularizer=l1l2(const_l1l2_regularization), b_regularizer=l1l2(const_l1l2_regularization)), input_shape=(const_frames_back, dimensions_state)))

        model.add(Flatten())

        model.add(Dense(500, activation='relu', W_regularizer=l1l2(const_l1l2_regularization), b_regularizer=l1l2(const_l1l2_regularization)))

    model.add(Dense(output_dim=dimensions_actions, W_regularizer=l1l2(const_l1l2_regularization),
                    activity_regularizer=activity_l1l2(const_l1l2_regularization)))

    optimizer = SGD(lr=const_lr)
    model.compile(loss='mse', optimizer=optimizer)

    logging.info('model build finished')

x = np.zeros((const_frames_back, dimensions_state)).tolist()
prev_action = 0

buffer = []

epoch = 0
episode = 0
total_epoch = 0
frames_total = 0
x = np.zeros((const_frames_back, dimensions_state)).tolist()

while True:
    episode += 1

    prev_frame = None
    prev_action = 0

    reward_total = 0
    reward_max = -100

    for frame in range(const_frames_max):
        frames_total += 1
        if env.game_over():
            break

        state = env.getGameState().tolist()
        state.append(prev_action + 1) #zero index = NAN

        if const_is_frame_diff:
            if frame == 0:
                state = np.zeros_like(state)
            else:
                state = state - prev_frame

        prev_frame = state

        x.append(state)
        x = x[-const_frames_back:]

        x_input = np.reshape(np.array(x), (1, const_frames_back, dimensions_state))
        q_values = model.predict(x_input, batch_size=1)
        q_values = np.reshape(q_values, (dimensions_actions,))

        action, q_values = get_action(q_values)

        reward = env.act(action) + 0.001
        reward_total += reward
        reward_max = max(reward_max, reward)

        buffer.append({
            'q' : q_values,
            'x' : x[:],
            'r' : reward
        })

        if(len(buffer) > const_memmory_buffer_max):
            buffer = buffer[-const_memmory_buffer_max:]

        if(len(buffer) > const_startup_frames):
            batch = random.sample(buffer, const_mini_batch_size)

            y_train = []
            x_train = []

            for i in range(len(batch)):
                y_index = np.argmax(batch[i]['q'])
                y_val = buffer[i]['q']

                y_val[y_index] = batch[i]['r']

                if i < len(batch) - 1:
                    x_next = batch[i + 1]['x']
                    x_input = np.reshape(np.array(x_next), (1, const_frames_back, dimensions_state))
                    q_values = model.predict(x_input, batch_size=1)
                    action, q_values = get_action(q_values)

                    y_val[y_index] = batch[i]['r'] + const_rl_gamma * np.max(q_values)

                x_train.append(batch[i]['x'])
                y_train.append(y_val)

            history = model.fit(np.array(x_train), np.array(y_train), batch_size=const_batch_size, nb_epoch=1, verbose=0)

            if frames_total % 10000 == 0:
                for loss in history.history['loss']:
                    total_epoch += 1
                    avg = 0
                    if reward_total > 0:
                        avg = reward_total / frame
                    file_csv_loss.write(
                        '{};{};{};{};{}\n'.format(total_epoch, episode, loss, reward_max, avg))
                    logging.info(
                        'epoch: {} episode: {} loss: {} max: {} avg: {}'.format(total_epoch, episode, loss, reward_max, avg))

                model.save('model-{}.h5'.format(const_version))

                file_csv_loss.flush()

        if const_is_playback:
            time.sleep(0.03)

    if const_is_playback:
        logging.info('game over')

    env.reset_game()

    epsilon_explore -= const_epsilon_explore_decay_step
    epsilon_explore = max(epsilon_explore, const_epsilon_explore_end)