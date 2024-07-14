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

    X = np.load('data/X_data.npy')
    y = np.load('data/y_data.npy')

    print("Finished loading data")
    print("======================================================")

    print("======================================================")
    print("Setting up neural network")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    inputs = tf.keras.Input(shape=(X.shape[1],))

    layers = []
    for _ in range(20):
        d1 = tf.keras.layers.Dense(32, activation='swish')(inputs)
        d2 = tf.keras.layers.Dense(16, activation='swish')(d1)

        d3 = tf.keras.layers.Dense(32, activation='swish')(inputs)
        d4 = tf.keras.layers.Dense(16, activation='swish')(d3)

        d6 = tf.keras.layers.Add()([d2, d4])
        d7 = tf.keras.layers.Multiply()([d2, d4])

        d8 = tf.keras.layers.Concatenate()([d2, d4, d6, d7])
        d9 = tf.keras.layers.Dense(32, activation='swish')(d8)
        d10 = tf.keras.layers.Dense(16, activation='swish')(d9)
        layers.append(d10)

    concat_layer = tf.keras.layers.Concatenate()(layers)
    c1 = tf.keras.layers.Dense(1024, activation='swish')(concat_layer)
    c2 = tf.keras.layers.Dense(256, activation='swish')(c1)
    c3 = tf.keras.layers.Dense(64, activation='swish')(c2)
    c4 = tf.keras.layers.Dense(32, activation='swish')(c3)
    c5 = tf.keras.layers.Dense(8, activation='swish')(c4)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(c5)

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
