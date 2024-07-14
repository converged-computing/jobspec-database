import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

from CustomCallback import MyThresholdCallback
from DataLoader import DataLoader, DataSet


def evaluate_model(train_data, test_data):
    verbose, epochs, batch_size = 1, 30, 32

    n_filters_1 = 8
    n_filters_2 = 8
    n_filters_3 = 80

    kernel_size_1 = 17
    kernel_size_2 = 3
    kernel_size_3 = 3

    drop_out_rate_1 = 0.0
    drop_out_rate_2 = 0.5
    drop_out_rate_3 = 0.3
    drop_out_rate_4 = 0.0

    pool_size_1 = 2
    pool_size_2 = 2
    pool_size_3 = 16

    dense_size_1 = 140

    with tf.distribute.MirroredStrategy().scope():
        model = keras.Sequential()

        model.add(layers.Conv2D(n_filters_1, (kernel_size_1, kernel_size_1), input_shape=(320, 320, 1), activation='relu', padding='same'))
        model.add(layers.Dropout(drop_out_rate_1))
        model.add(layers.MaxPool2D((pool_size_1, pool_size_1), padding='same'))

        model.add(layers.Conv2D(n_filters_2, (kernel_size_2, kernel_size_2), activation='relu', padding='same'))
        model.add(layers.Dropout(drop_out_rate_2))
        model.add(layers.MaxPool2D((pool_size_2, pool_size_2), padding='same'))

        model.add(layers.Conv2D(n_filters_3, (kernel_size_3, kernel_size_3), activation='relu', padding='same'))
        model.add(layers.Dropout(drop_out_rate_3))
        model.add(layers.MaxPool2D((pool_size_3, pool_size_3), padding='same'))

        model.add(layers.Flatten())
        model.add(layers.Dense(dense_size_1, activation='relu'))
        model.add(layers.Dropout(drop_out_rate_4))
        model.add(layers.Dense(8, activation='softmax'))


        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network

    callback1 = keras.callbacks.EarlyStopping(monitor='val_loss',
                                              min_delta=1e-4,
                                              patience=2,
                                              verbose=1, mode='auto')

    callback2 = MyThresholdCallback(threshold=1.0)

    history = model.fit(train_data, epochs=epochs, verbose=verbose, validation_data=test_data, callbacks=[callback1, callback2])
    return np.max(history.history['val_accuracy'])

def k_fold_reading():
    accuracies = []
    for fold in range(0, 10):

        subs = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
        ]

        resolution = 320
        thickness = 1
        gradient = True

        loader = DataLoader(DataSet.SEDENTARY)
        trainX, trainy, testX, testy = loader.load_1D(framelength=1024, testsubjects=subs[fold])
        trainX, testX = loader.transform_to_2d(trainX, resolution, thickness, gradient), loader.transform_to_2d(testX, resolution, thickness, gradient)

        train_data = tf.data.Dataset.from_tensor_slices((trainX, trainy)).batch(batch_size=128)
        test_data = tf.data.Dataset.from_tensor_slices((testX, testy)).batch(batch_size=128)

        accuracy = evaluate_model(train_data, test_data)
        print(accuracy)
        accuracies.append(accuracy)

    print(f'avrg acc = {np.mean(accuracies)}')

def k_fold_reading_frame():

    resolution = 320
    thickness = 1
    gradient = True

    loader = DataLoader(DataSet.SEDENTARY)
    allX, ally, blaX, blay = loader.load_1D(framelength=1024, testsubjects=[], allTrain=True)
    allX = loader.transform_to_2d(allX, resolution, thickness, gradient)
    idxes = np.arange(len(allX))
    np.random.shuffle(idxes)
    allX = allX[idxes]
    ally = ally[idxes]

    accuracies = []
    for fold in range(0, 10):
        foldSize = np.round(len(allX) / 10).astype(np.int32)
        test_mask = np.zeros(len(allX), dtype=bool)
        test_mask[fold*foldSize : min(((fold+1)*foldSize), len(test_mask))] = True
        trainX = allX[~test_mask]
        trainy = ally[~test_mask]
        testX = allX[test_mask]
        testy = ally[test_mask]

        train_data = tf.data.Dataset.from_tensor_slices((trainX, trainy)).batch(batch_size=128)
        test_data = tf.data.Dataset.from_tensor_slices((testX, testy)).batch(batch_size=128)

        accuracy = evaluate_model(train_data, test_data)
        print(accuracy)
        accuracies.append(accuracy)

    print(f'avrg acc = {np.mean(accuracies)}')

k_fold_reading_frame()
k_fold_reading()