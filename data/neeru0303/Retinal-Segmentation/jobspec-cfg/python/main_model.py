#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np
import tensorflow as tf
import json
import keras

from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Flatten, Dropout, Input, concatenate, merge, Add, Dropout
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D, ZeroPadding2D, Activation
from keras.layers import MaxPooling2D, UpSampling2D, Permute
from keras import backend as K
from keras.activations import softmax
import keras.backend.tensorflow_backend as tfb
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import SGD,Adam
from time import time
from base_model import(BaseModel,sigmoid_cross_entropy_with_logits,
                        image_accuracy,softmax_cross_entropy_with_logits)



def parse_args():
    """
        function for argument parsing
        :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", "-c", help="Cache data wherever possible", action='store_true')
    parser.add_argument("--classification", "-t", help="Cache data wherever possible",
                        default=4, type=int)
    parser.add_argument("--dataset", "-d", help="dataset small or big",
                         default="big", choices=["small", "big"], type=str)
    parser.add_argument("--reload", "-r", help="reload data", action='store_true')
    parser.add_argument("--activation", "-a", help="activation function for conv layers",
                         default="relu")
    parser.add_argument("--log_level", "-l", help="Set loglevel for debugging and analysis",
                         default="INFO")
    args = parser.parse_args()
    return args


class RetinaModel(BaseModel):
    def __init__(self, classification=3, dataset="big", reload=False, activation='relu', cache=True):
        super(RetinaModel, self).__init__(classification, dataset, reload, activation, cache)

    def create_model(self):
        print(self.activation)
        input_shape =(3, 565, 565)

        data_input = Input(shape=input_shape, name="data_input")
        conv1_1 = Conv2D(64, kernel_size=(3, 3), activation=self.activation, name="conv1_1",
                          padding="SAME")(data_input)
        conv1_1 = Dropout(0.2, name="Drop1_1")(conv1_1)
        conv1_2 = Conv2D(64, kernel_size=(3, 3), activation=self.activation, name="conv1_2",
                          padding="SAME")(conv1_1)
        conv1_2 = Dropout(0.2, name="Drop1_2")(conv1_2)
        max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pool1',
                                  padding="SAME")(conv1_2)

        # Convolution Layer 2
        conv2_1 = Conv2D(128, kernel_size=(3, 3), activation=self.activation, name="conv2_1",
                          padding="SAME")(max_pool1)
        conv2_1 = Dropout(0.2, name="Drop2_1")(conv2_1)
        conv2_2 = Conv2D(128, kernel_size=(3, 3), activation=self.activation, name="conv2_2",
                          padding="SAME")(conv2_1)
        conv2_2 = Dropout(0.2, name="Drop2_2")(conv2_2)
        max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pool2',
                                  padding="SAME")(conv2_2)

        # Convolution Layer3
        conv3_1 = Conv2D(256, kernel_size=(3, 3), activation=self.activation, name="conv3_1",
                          padding="SAME")(max_pool2)
        conv3_1 = Dropout(0.2, name="Drop3_1")(conv3_1)
        conv3_2 = Conv2D(256, kernel_size=(3, 3), activation=self.activation, name="conv3_2",
                          padding="SAME")(conv3_1)
        conv3_2 = Dropout(0.2, name="Drop3_2")(conv3_2)
        conv3_3 = Conv2D(256, kernel_size=(3, 3), activation=self.activation, name="conv3_3",
                          padding="SAME")(conv3_2)
        conv3_3 = Dropout(0.2, name="Drop3_3")(conv3_3)
        max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pool3',
                                  padding="SAME")(conv3_3)

        # Convolution Layer4
        conv4_1 = Conv2D(512, kernel_size=(3, 3), activation=self.activation, name="conv4_1",
                          padding="SAME")(max_pool3)
        conv4_1 = Dropout(0.2, name="Drop4_1")(conv4_1)
        conv4_2 = Conv2D(512, kernel_size=(3, 3), activation=self.activation, name="conv4_2",
                          padding="SAME")(conv4_1)
        conv4_2 = Dropout(0.2, name="Drop4_2")(conv4_2)
        conv4_3 = Conv2D(512, kernel_size=(3, 3), activation=self.activation, name="conv4_3",
                          padding="SAME")(conv4_2)
        conv4_3 = Dropout(0.2, name="Drop4_3")(conv4_3)

        #
        conv1_2_16 = Conv2D(16, kernel_size=(3, 3), name="conv1_2_16",
                             padding="SAME")(conv1_2)
        conv1_2_16 = Dropout(0.2, name="Drop1_2_16")(conv1_2_16)
        conv2_2_16 = Conv2D(16, kernel_size=(3, 3), name="conv2_2_16",
                             padding="SAME")(conv2_2)
        conv2_2_16 = Dropout(0.2, name="Drop2_2_16")(conv2_2_16)
        conv3_3_16 = Conv2D(16, kernel_size=(3, 3), name="conv3_3_16",
                             padding="SAME")(conv3_3)
        conv3_3_16 = Dropout(0.2, name="Drop3_3_16")(conv3_3_16)
        conv4_3_16 = Conv2D(16, kernel_size=(3, 3), name="conv4_3_16",
                             padding="SAME")(conv4_3)
        conv4_3_16 = Dropout(0.2, name="Drop4_3_16")(conv4_3_16)

        # Deconvolution Layer1
        side_multi2_up = UpSampling2D(size=(2, 2), name="side_multi2_up")(conv2_2_16)

        upside_multi2 = Cropping2D(cropping=((0, 1),(0, 1)), name="upside_multi2")(side_multi2_up)

        #Decovolution Layer2
        side_multi3_up = UpSampling2D(size=(4, 4), name="side_multi3_up")(conv3_3_16)
        upside_multi3 = Cropping2D(cropping=((1, 2),(1, 2)), name="upside_multi3")(side_multi3_up)

        # Deconvolution Layer3
        side_multi4_up = UpSampling2D(size=(8, 8), name="side_multi4_up")(conv4_3_16)
        upside_multi4 = Cropping2D(cropping=((1, 2),(1, 2)), name="upside_multi4")(side_multi4_up)

        # Specialized Layer
        concat_upscore = concatenate([conv1_2_16, upside_multi2, upside_multi3, upside_multi4],
                                      name="concat-upscore", axis=1)
        upscore_fuse = Conv2D(self._classification, kernel_size=(1, 1), name="upscore_fuse")(concat_upscore)
        upscore_fuse = Dropout(0.2, name="Dropout_Classifier")(upscore_fuse)
        upscore_fuse = Activation('sigmoid')(upscore_fuse)
        self.model = Model(inputs=[data_input], outputs=[upscore_fuse])


    def set_weights(self):
        if self.cache and os.path.exists("cache/keras_crop_model_weights_4class_reg.h5"):
            print("yes")
            # self.model.load_weights("cache/keras_crop_model_weights_4class_reg.h5")
            with open("cache/main_model.json") as f:
                dev_model = model_from_json(json.dumps(json.load(f)))
            dev_model.load_weights("cache/keras_crop_model_weights_4class_reg.h5")

            for dev_layer, layer in zip(dev_model.layers, self.model.layers):
                try:
                    layer.set_weights(dev_layer.get_weights())
                except:
                    print(layer.name)
    
    def fit(self):
        print(self.train_images.shape)
        sgd = SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=True)
        weight_save_callback = keras.callbacks.ModelCheckpoint('/cache/checkpoint_weights.h5', monitor='val_loss',
                                                verbose=0, save_best_only=True, mode='auto')
        tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph/{}/'.format(time()), histogram_freq=20,
                                     write_graph=True, write_images=False)

        self.model.compile(optimizer=sgd, loss=sigmoid_cross_entropy_with_logits,
                            metrics=['accuracy', image_accuracy])

        self.model.fit(self.train_images, self.train_labels, batch_size=5, epochs=2000,
                        callbacks=[tb_callback], validation_split=0.05, verbose=1)

        self.model.save_weights(os.path.join('cache', 
                                             'keras_crop_model_weights_4class_reg_{}.h5'.format(self.activation)))
    
    def predict(self, data):
        test_predict = self.model.predict(data, batch_size=10)
        print(test_predict[0])
        print(test_predict.shape)
        np.save('cache/test_predict2_class_4_{}.npy'.format(self.activation), test_predict)


if __name__ == '__main__':
    args = parse_args()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "1"
    set_session(tf.Session(config=config))
    rm = RetinaModel(classification=args.classification, dataset=args.dataset,
                      reload=args.reload, activation=args.activation, cache=args.cache)
    rm.create_model()
    rm.set_weights()
    rm.get_data()
    print(rm.test_labels.shape)
    print(rm.train_images.shape)
    rm.run()
    # rm.predict(data = rm.test_images)
    K.clear_session()