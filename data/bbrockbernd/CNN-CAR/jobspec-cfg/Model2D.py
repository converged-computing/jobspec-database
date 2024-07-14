# import keras_tuner
import numpy as np
import pandas as pd
import cv2
# from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.layers as layers
import keras_tuner as kt
from time import time

from CustomOptimizer import CustomOptimizer
from DataLoader import DataLoader, DataSet


def load_desktop_data_1D(testPerson = 8, framelength = 256):

    testX = np.zeros((0, framelength, 2))
    testy = np.zeros((0, 6))

    trainX = np.zeros((0, framelength, 2))
    trainy = np.zeros((0, 6))
    for i in range(1, 9):
        for class_index, activity in enumerate(['BROWSE', 'PLAY', 'READ', 'SEARCH', 'WATCH', 'WRITE']):
            raw_data = pd.read_csv(f'DesktopActivityOld/P{i}/P{i}_{activity}.csv', header=None).values

            #cut data into frames
            indices = np.array([np.arange(0, 9000, framelength/2)[:-2], np.arange(framelength, 9000, framelength/2)]).astype(int)
            frames = []
            for ind in indices.T:
                frames.append(raw_data[ind[0] : ind[1], :])

            framesX = np.array(frames)
            classy = np.zeros(6)
            classy[class_index] = 1
            framesy = np.tile(classy, (framesX.shape[0], 1))

            if testPerson == i:
                testX = np.concatenate((testX, framesX))
                testy = np.concatenate((testy, framesy))
            else:
                trainX = np.concatenate((trainX, framesX))
                trainy = np.concatenate((trainy, framesy))

    return trainX, trainy, testX, testy

# data = (n, framelen, 2)
def transform_to_2d(data, resolution = 500):

    shifted = data - np.min(data)
    data = shifted / np.max(shifted)
    data2d = np.zeros((data.shape[0], resolution, resolution, 3), dtype=np.uint8)
    for i in range(data.shape[0]):
        current = data[i]
        current = np.floor(current * resolution).astype(int)
        current[current == resolution] = resolution - 1
        cv2.polylines(data2d[i], np.int32([current]), False, color=(255, 255, 255), thickness=1)

    data2dbw = np.zeros((data2d.shape[0], data2d.shape[1], data2d.shape[2]))
    data2dbw[data2d[:, :, :, 0] == 255] = 1
    return data2dbw[:, :, :, np.newaxis]

def create_moddel(strategy = None):
    # TODO leaky relu


    model = keras.Sequential()

    model.add(layers.Conv2D(16, (5, 5), input_shape=(128, 128, 1), activation='relu', padding='same'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.ReLU())
    # model.add(layers.Dropout(0.2))
    model.add(layers.MaxPool2D((4, 4)))

    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.ReLU())
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPool2D((4, 4)))

    model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.ReLU())
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPool2D((4, 4)))

    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.ReLU())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(8, activation='softmax'))


    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def run_this_mofo():
    loader = DataLoader(DataSet.SEDENTARY)
    trainX, trainy, testX, testy = loader.load_1D(validation=0.2, framelength=512)
    # trainX, trainy, testX, testy = load_desktop_data_1D()

    trainX = loader.transform_to_2d(trainX, 128)
    testX = loader.transform_to_2d(testX, 128)

    start = time()

    # strategy = tf.distribute.MirroredStrategy()
    # print(f'Number of devices: {strategy.num_replicas_in_sync}')

    train_data = tf.data.Dataset.from_tensor_slices((trainX, trainy)).batch(batch_size=32)
    test_data = tf.data.Dataset.from_tensor_slices((testX, testy)).batch(batch_size=32)


    model = create_moddel()
    verbose, epochs, batch_size = 1, 10, 32
    model.fit(train_data, epochs=epochs, verbose=verbose)
    _, accuracy = model.evaluate(test_data, verbose=verbose)
    print(accuracy)

    end = time()

    print(f'duration: {end - start}')


    # for i in range(0, len(joe), 10):
    #     plt.imshow(joe[i])
    #     plt.title(np.array(['BROWSE', 'PLAY', 'READ', 'SEARCH', 'WATCH', 'WRITE'])[testy[i].astype(bool)])
    #     plt.savefig(f'figures/{i}')


class HyperModel2D(kt.HyperModel):

    def build(self, hp):
        resolution = hp.Int('resolution', 64, 320, 64)

        n_filters_1 = hp.Int('filter count convlayer 1', 8, 128, 8)
        n_filters_2 = hp.Int('filter count convlayer 2', 8, 128, 8)
        n_filters_3 = hp.Int('filter count convlayer 3', 8, 128, 8)

        kernel_size_1 = hp.Int('kernel size convlayer 1', 3, 15, 2)
        kernel_size_2 = hp.Int('kernel size convlayer 2', 3, 15, 2)
        kernel_size_3 = hp.Int('kernel size convlayer 3', 3, 15, 2)

        drop_out_rate_1 = hp.Float('drop out rate 1', 0.0, 0.5, 0.1)
        drop_out_rate_2 = hp.Float('drop out rate 2', 0.0, 0.5, 0.1)
        drop_out_rate_3 = hp.Float('drop out rate 3', 0.0, 0.5, 0.1)
        drop_out_rate_4 = hp.Float('drop out rate 4', 0.0, 0.5, 0.1)

        pool_size_1 = hp.Choice('pool size 1', [2, 4, 8])
        pool_size_2 = hp.Choice('pool size 2', [2, 4, 8])
        pool_size_3 = hp.Choice('pool size 3', [2, 4, 8])

        dense_size_1 = hp.Int('dense size 2', 20, 140, 20)

        with tf.distribute.MirroredStrategy().scope():

            model = keras.Sequential()

            model.add(layers.Conv2D(n_filters_1, (kernel_size_1, kernel_size_1), input_shape=(resolution, resolution, 1), activation='relu', padding='same'))
            # model.add(layers.BatchNormalization())
            # model.add(layers.ReLU())
            model.add(layers.Dropout(drop_out_rate_1))
            model.add(layers.MaxPool2D((pool_size_1, pool_size_1), padding='same'))

            model.add(layers.Conv2D(n_filters_2, (kernel_size_2, kernel_size_2), activation='relu', padding='same'))
            # model.add(layers.BatchNormalization())
            # model.add(layers.ReLU())
            model.add(layers.Dropout(drop_out_rate_2))
            model.add(layers.MaxPool2D((pool_size_2, pool_size_2), padding='same'))

            model.add(layers.Conv2D(n_filters_3, (kernel_size_3, kernel_size_3), activation='relu', padding='same'))
            # model.add(layers.BatchNormalization())
            # model.add(layers.ReLU())
            model.add(layers.Dropout(drop_out_rate_3))
            model.add(layers.MaxPool2D((pool_size_3, pool_size_3), padding='same'))

            model.add(layers.Flatten())
            model.add(layers.Dense(dense_size_1, activation='relu'))
            # model.add(layers.BatchNormalization())
            # model.add(layers.ReLU())
            model.add(layers.Dropout(drop_out_rate_4))
            model.add(layers.Dense(6, activation='softmax'))


            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self, hp, model, *args, **kwargs):
        framelength = hp.Int('frame length', 512, 4096, 64)
        resolution = hp.get('resolution')
        thickness = hp.Int('thickness', 1, 4, 1)
        gradient = hp.Boolean('Gradient')


        trainX, trainy, testX, testy = DataLoader(DataSet.READING).load_1D(framelength=framelength)
        trainX = DataLoader(DataSet.READING).transform_to_2d(trainX, resolution, thickness=thickness, gradient=gradient)
        testX = DataLoader(DataSet.READING).transform_to_2d(testX, resolution, thickness=thickness)

        verbose, epochs, batch_size = 0, 10, 32

        train_data = tf.data.Dataset.from_tensor_slices((trainX, trainy)).batch(batch_size=128)
        test_data = tf.data.Dataset.from_tensor_slices((testX, testy)).batch(batch_size=128)

        return model.fit(train_data, *args, validation_data=test_data, **kwargs)


def optimize():

    # tuner = kt.RandomSearch(
    #     HyperModel2D(),
    #     objective="val_accuracy",
    #     max_trials=3,
    #     overwrite=True,
    #     directory="my_dir",
    #     project_name="tune_hypermodel",
    # )
    # tuner.search(epochs=5)

    # tuner = kt.Hyperband(HyperModel2D(),
    #                      objective='val_accuracy',
    #                      max_epochs=15,
    #                      factor=3,
    #                      directory='tuning',
    #                      project_name='tune_hypermodel')
    # tuner.search()

    # tuner = kt.RandomSearch(
    #     HyperModel2D(),
    #     objective="val_accuracy",
    #     max_trials=3,
    #     overwrite=True,
    #     directory="my_dir",
    #     project_name="tune_hypermodel",
    # )
    # tuner.search(epochs=5)


    tuner = CustomOptimizer(HyperModel2D(),
                            objective="val_accuracy",
                            max_trials=200,
                            overwrite=False,
                            directory="/scratch/bbrockbernd/tuning_reading_2d",
                            project_name="tune_hypermodel",
                            executions_per_trial=3)

    callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=1e-4,
                                             patience=2,
                                             verbose=0, mode='auto')

    tuner.search(epochs=30, callbacks=[callback])

optimize()
# run_this_mofo()