from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.layers as layers
import keras_tuner as kt

from CustomOptimizer import CustomOptimizer
from DataLoader import DataLoader, DataSet


class HyperModel2D(kt.HyperModel):

    def build(self, hp):
        resolution = hp.Int('resolution', 64, 320, 64)

        n_filters_1 = hp.Int('filter count convlayer 1', 8, 128, 8)
        n_filters_2 = hp.Int('filter count convlayer 2', 8, 128, 8)
        n_filters_3 = hp.Int('filter count convlayer 3', 8, 128, 8)

        kernel_size_1 = hp.Int('kernel size convlayer 1', 3, 17, 2)
        kernel_size_2 = hp.Int('kernel size convlayer 2', 3, 17, 2)
        kernel_size_3 = hp.Int('kernel size convlayer 3', 3, 17, 2)

        drop_out_rate_1 = hp.Float('drop out rate 1', 0.0, 0.5, 0.1)
        drop_out_rate_2 = hp.Float('drop out rate 2', 0.0, 0.5, 0.1)
        drop_out_rate_3 = hp.Float('drop out rate 3', 0.0, 0.5, 0.1)
        drop_out_rate_4 = hp.Float('drop out rate 4', 0.0, 0.5, 0.1)

        pool_size_1 = hp.Choice('pool size 1', [2, 4, 8, 16])
        pool_size_2 = hp.Choice('pool size 2', [2, 4, 8, 16])
        pool_size_3 = hp.Choice('pool size 3', [2, 4, 8, 16])

        dense_size_1 = hp.Int('dense size 2', 20, 140, 20)

        with tf.distribute.MirroredStrategy().scope():

            model = keras.Sequential()

            model.add(layers.Conv2D(n_filters_1, (kernel_size_1, kernel_size_1), input_shape=(resolution, resolution, 1), activation='relu', padding='same'))
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
        return model

    def fit(self, hp, model, *args, **kwargs):
        framelength = hp.Int('frame length', 256, 1024, 64)
        resolution = hp.get('resolution')
        thickness = hp.Int('thickness', 1, 4, 1)
        gradient = hp.Boolean('Gradient')

        loader = DataLoader(DataSet.SEDENTARY)

        trainX, trainy, testX, testy = loader.load_1D(framelength=framelength, validation=0.2)
        trainX = loader.transform_to_2d(trainX, resolution, thickness=thickness, gradient=gradient)
        testX = loader.transform_to_2d(testX, resolution, thickness=thickness)

        train_data = tf.data.Dataset.from_tensor_slices((trainX, trainy)).batch(batch_size=128)
        test_data = tf.data.Dataset.from_tensor_slices((testX, testy)).batch(batch_size=128)

        return model.fit(train_data, *args, validation_data=test_data, **kwargs)


def optimize():
    tuner = CustomOptimizer(HyperModel2D(),
                            objective="val_accuracy",
                            max_trials=200,
                            overwrite=False,
                            directory="/scratch/bbrockbernd/tuning_2D_sedentary_subject",
                            project_name="tune_hypermodel",
                            executions_per_trial=3)

    callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=1e-4,
                                             patience=2,
                                             verbose=0, mode='auto')

    tuner.search(epochs=30, callbacks=[callback])

optimize()
