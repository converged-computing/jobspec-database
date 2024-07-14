'''
PS#2
Q3 Inception Module for CIFAR-10 dataset

'''
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.layers import Input
from keras.utils import np_utils
from keras.datasets import cifar10

epochs = 100

# Get the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Get the data ready
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Create imput
input_img = Input(shape=(32, 32, 3))


# Create Volumes for the Inception module
volume_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
volume_1 = Dropout(0.75)(volume_1)
volume_1 = BatchNormalization()(volume_1)

volume_2 = Conv2D(96, (1, 1), padding='same', activation='relu')(input_img)
volume_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(volume_2)
volume_2 = Dropout(0.75)(volume_2)
volume_2 = BatchNormalization()(volume_2)

volume_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(input_img)
volume_3 = Conv2D(32, (5, 5), padding='same', activation='relu')(volume_3)
volume_3 = Dropout(0.75)(volume_3)
volume_3 = BatchNormalization()(volume_3)

volume_4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
volume_4 = Conv2D(32, (1, 1), padding='same', activation='relu')(volume_4)
volume_4 = Dropout(0.75)(volume_4)
volume_4 = BatchNormalization()(volume_4)

# Concatenate all volumes of the Inception module
inception_module = keras.layers.concatenate([volume_1, volume_2, volume_3,
                                             volume_4], axis=3)
output = Flatten()(inception_module)
out = Dense(10, activation='softmax')(output)


model = Model(inputs=input_img, outputs=out)
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, y_train, validation_data=(
    X_test, y_test), shuffle=True, epochs=epochs, batch_size=512)


scores = model.evaluate(X_test, y_test, verbose=0)
print("Q3 Accuracy: %.2f%%" % (scores[1]*100))
