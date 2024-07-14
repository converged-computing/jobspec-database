import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("WARNING: NO GPUs FOUND!!!!!!!!!!!!!!!!!!!!!")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    print("======================================================")
    print("Loading data")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    X = np.load('data/X_data.npy')[:, -150:] # Less number of columns
    y = np.load('data/y_data.npy')

    print("Finished loading data")
    print("======================================================")

    print("======================================================")
    print("Setting up neural network")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # simpler model
    inputs = tf.keras.Input(shape=(X.shape[1],))

    d1 = tf.keras.layers.Dense(16, activation='swish')(inputs)
    d2 = tf.keras.layers.Dense(8, activation='swish')(d1)

    d3 = tf.keras.layers.Dense(16, activation='swish')(inputs)
    d4 = tf.keras.layers.Dense(8, activation='swish')(d3)

    d6 = tf.keras.layers.Add()([d2, d4])
    d7 = tf.keras.layers.Multiply()([d2, d4])

    d8 = tf.keras.layers.Concatenate()([d2, d4, d6, d7])
    d9 = tf.keras.layers.Dense(16, activation='swish')(d8)
    d10 = tf.keras.layers.Dense(8, activation='swish')(d9)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(d10)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])


    print("Finished Setting up neural network")
    print("======================================================")


    print("======================================================")
    print("Start training model")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    model.fit(X, y, epochs=5, batch_size=128, validation_split=0.25)

    print("Finished training model")
    print("======================================================")
