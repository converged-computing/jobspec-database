'''
PS#2
Q1 Data augumentation for the CNN for CIFAR-10 dataset

'''
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import pandas

batch_size = 256
num_classes = 10
epochs = 50

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# One-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Feature normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Keras Model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])


''' Your task
Please use Keras ImageDataGenerator to perfrom image data augumentation.
Hint: You can use (https://keras.io/preprocessing/image/#imagedatagenerator)
'''
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    zoom_range=0.25,
    shear_range=0.2,
    width_shift_range=0.5,
    height_shift_range=0.5,
    horizontal_flip=True
)


history = model.fit_generator(datagen.flow(
    x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) / 32, epochs=epochs, validation_data=(
    x_test, y_test))

pandas.DataFrame(history.history).to_csv("q1.history.csv")
