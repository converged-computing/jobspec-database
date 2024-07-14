'''
PS#2
Q2 ResNet Implementation for CIFAR-10 dataset

'''

import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
import pandas

# Network depth
# Please explore depths of [20, 32, 44, 56]

depths = [20, 32, 44, 56]
skips = [False, True]

for skip in skips:
    for depth in depths:

        # Training parameters
        batch_size = 128
        epochs = 200
        num_classes = 10

        model_type = 'ResNet%d' % (depth)

        # Load the CIFAR10 data.
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # Input image dimensions.
        input_shape = x_train.shape[1:]

        # Normalize data.
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # Subtract mean
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

        # Print data info
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print('y_train shape:', y_train.shape)

        # One-hot encoding.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)


        # Learning rate scheduler
        # lr to be reduced based on number of epochs
        def lr_schedule(epoch):

            lr = 1e-3
            if epoch > 180:
                lr *= 0.5e-3
            elif epoch > 150:
                lr *= 1e-3
            elif epoch > 100:
                lr *= 1e-2
            elif epoch > 50:
                lr *= 1e-1
            print('Learning rate: ', lr)
            return lr

        # A function to build layers for the Resnet:
            # 1. Conv
            # 2. Batch normalization
            # 3. Activation


        def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
            """2D Convolution-Batch Normalization-Activation stack builder

            # Arguments
                inputs (tensor): input tensor from input image or previous layer
                num_filters (int): Conv2D number of filters
                kernel_size (int): Conv2D square kernel dimensions
                strides (int): Conv2D square stride dimensions
                activation (string): activation name
                batch_normalization (bool): whether to include batch normalization

            # Returns
                x (tensor): tensor as input to the next layer
            """
            # Convolution operation
            conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides,
                        padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))

            x = inputs
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            return x


        def resnet_create(input_shape, depth, num_classes=10):
            """
            First stack does not change the size
            Later, at the beginning of each stack, the feature map size is halved (downsampled)
            by a convolutional layer with strides=2, while the number of filters is
            doubled. Within each stage, the layers have the same number filters and the
            same number of filters.
            Features maps sizes:
            stack 0: 32x32, 16
            stack 1: 16x16, 32
            stack 2:  8x8,  64

            # Arguments
                input_shape (tensor): shape of input image tensor
                depth (int): number of core convolutional layers
                num_classes (int): number of classes (CIFAR10 has 10)

            # Returns
                model (Model): Keras model instance
            """
            if (depth - 2) % 6 != 0:
                raise ValueError('depth should be 6n+2 (eg 20, 32, 44)')
            # Start model definition.
            num_filters = 16
            num_res_blocks = int((depth - 2) / 6)

            inputs = Input(shape=input_shape)
            x = resnet_layer(inputs=inputs)
            # Instantiate the stack of residual units
            for stack in range(3):
                for res_block in range(num_res_blocks):
                    strides = 1
                    if stack > 0 and res_block == 0:  # first layer but not first stack
                        strides = 2  # downsample
                    y = resnet_layer(
                        inputs=x, num_filters=num_filters, strides=strides)
                    y = resnet_layer(
                        inputs=y, num_filters=num_filters, activation=None)
                    if stack > 0 and res_block == 0:  # first layer but not first stack
                        # linear projection residual shortcut connection to match changed dims
                        x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1,
                                        strides=strides, activation=None, batch_normalization=False)
                    # Add skip connection

                    if skip == True:
                        x = keras.layers.add([x, y])
                        x = Activation('relu')(x)
                num_filters *= 2                    # Increase number of filter

            # Add classifier on top.
            x = AveragePooling2D(pool_size=8)(x)
            y = Flatten()(x)
            outputs = Dense(num_classes, activation='softmax',
                            kernel_initializer='he_normal')(y)

            # Instantiate model.
            model = Model(inputs=inputs, outputs=outputs)
            return model


        model = resnet_create(input_shape=input_shape, depth=depth)

        model.compile(loss='categorical_crossentropy', optimizer=Adam(
            lr=lr_schedule(0)), metrics=['accuracy'])
        model.summary()
        print(model_type)

        # Prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        # Prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = ModelCheckpoint(
            filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)

        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(
            0.1), cooldown=0, patience=5, min_lr=0.5e-6)

        callbacks = [checkpoint, lr_reducer, lr_scheduler]

        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                            validation_data=(x_test, y_test), shuffle=True, callbacks=callbacks)


        pandas.DataFrame(history.history).to_csv(f"q2.history.{depth}.{skip}.csv")
