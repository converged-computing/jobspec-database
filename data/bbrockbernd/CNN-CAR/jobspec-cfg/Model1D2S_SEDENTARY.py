import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from CustomOptimizer import CustomOptimizer
from DataLoader import DataLoader, DataSet
import keras_tuner as kt


class HyperModel1D(kt.HyperModel):

    def build(self, hp):
        frame_length = hp.Int('frame length', 256, 4096, 64)

        n_filters_1 = hp.Int('filter count 1', 32, 256, 8)
        kernel_size_1 = hp.Int('kernel size 1', 3, 25, 2)
        drop_1 = hp.Float('drop out 1', 0.0, 0.5, 0.1)
        pool_1 = hp.Choice('max pool 1', [2, 4, 8, 16])

        second_convolution = hp.Boolean('Enable 2nd Conv')

        dense_size_1 = hp.Int('Dense size 1', 60, 340, 20)
        drop_3 = hp.Float('drop out 3', 0.0, 0.5, 0.1)

        second_dense = hp.Boolean('Enable 2nd Dense')



        model = keras.Sequential()

        model.add(layers.Conv1D(filters=n_filters_1, kernel_size=kernel_size_1, activation='relu', input_shape=(frame_length, 2), padding='same'))
        model.add(layers.Dropout(drop_1))
        model.add(layers.MaxPooling1D(pool_size=pool_1))

        if second_convolution:
            with hp.conditional_scope('Enable 2nd Conv', [True]):
                n_filters_2 = hp.Int('filter count 2', 32, 256, 8)
                kernel_size_2 = hp.Int('kernel size 2', 3, 25, 2)
                drop_2 = hp.Float('drop out 2', 0.0, 0.5, 0.1)
                pool_2 = hp.Choice('max pool 2', [2, 4, 8, 16, 32, 64])

                model.add(layers.Conv1D(filters=n_filters_2, kernel_size=kernel_size_2, activation='relu', padding='same'))
                model.add(layers.Dropout(drop_2))
                model.add(layers.MaxPooling1D(pool_size=pool_2))

        model.add(layers.Flatten())
        model.add(layers.Dense(dense_size_1, activation='relu'))
        model.add(layers.Dropout(drop_3))

        if second_dense:
            with hp.conditional_scope('Enable 2nd Dense', [True]):
                dense_size_2 = hp.Int('Dense size 2', 60, 200, 20)
                drop_4 = hp.Float('drop out 4', 0.0, 0.8, 0.2)

                model.add(layers.Dense(dense_size_2, activation='relu'))
                model.add(layers.Dropout(drop_4))

        model.add(layers.Dense(8, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def fit(self, hp, model, *args, **kwargs):
        loader = DataLoader(DataSet.SEDENTARY)
        frame_length = hp.get('frame length')
        trainX, trainy, testX, testy = loader.load_1D_framesplit(framelength=frame_length, validation=0.2)

        return model.fit(trainX, trainy, *args, batch_size=256, validation_data=(testX, testy), **kwargs)



def tune_params():
    tuner = CustomOptimizer(HyperModel1D(),
                            objective="val_accuracy",
                            max_trials=300,
                            overwrite=False,
                            directory="/scratch/bbrockbernd/tuning_1D_sedentary_frame",
                            project_name="tune_hypermodel",
                            executions_per_trial=3)

    callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=1e-4,
                                             patience=2,
                                             verbose=0, mode='auto')

    tuner.search(epochs=30, callbacks=[callback])


tune_params()