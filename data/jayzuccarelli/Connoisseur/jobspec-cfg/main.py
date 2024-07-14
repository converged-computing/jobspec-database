import numpy as np
import os as os
import pathlib as pl
import keras as ks
from PIL import ImageFile


global USER, NUM_GPUS
USER = 'killianf'
NUM_GPUS = 2

def preprocess_input():




def main():
    # Locations
    LOCATION = '/pool001/' + USER + '/Connoisseur/Artworks'
    PATH = pl.Path(LOCATION)

    # Data
    IMAGES = PATH
    CLASSES = [item.name for item in PATH.glob('*')]

    # Parameters
    BATCH_SIZE = 128
    EPOCHS = 1000
    STEPS_PER_EPOCH = np.ceil(len(CLASSES) / BATCH_SIZE)
    IMG_WIDTH = 256
    IMG_HEIGHT = 256


    # Build Datasets
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    ks.backend.set_image_dim_ordering('tf')
    image_generator = ks.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                rotation_range=20,
                                                                channel_shift_range=20,
                                                                horizontal_flip=True)
    train_data_gen = image_generator.flow_from_directory(directory=LOCATION,
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         interpolation='lanczos:random',
                                                         classes=CLASSES)


    # Model
    inputs = ks.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)) #TODO: Try None, None - Should it be width and height or viceversa?
    step_1 = ks.layers.Convolution2D(filters=32, kernel_size=(3, 3), activation='relu',
                                     padding='same', data_format="channels_last")(inputs)
    step_2 = ks.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(step_1)
    step_3 = ks.layers.Convolution2D(filters=64, kernel_size=(5, 5), activation='relu',
                                     padding='same', data_format="channels_last")(step_2)
    step_4 = ks.layers.MaxPooling2D(pool_size=(2, 2), padding='same', strides=2)(step_3)
    step_5 = ks.layers.Flatten()(step_4)
    step_6 = ks.layers.Dense(256, activation='relu')(step_5)
    step_7 = ks.layers.Dense(128, activation='relu')(step_6)
    step_8 = ks.layers.Dropout(rate=0.5)(step_7)
    output = ks.layers.Dense(len(CLASSES), activation='softmax')(step_8)

    model = ks.models.Model(inputs=inputs, outputs=output)

    model = ks.utils.multi_gpu_model(model, gpus=NUM_GPUS)

    optimizer = ks.optimizers.Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

    model.fit_generator(train_data_gen, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, verbose=1)

    if ks.backend.backend() == 'tensorflow':
        ks.backend.clear_session()

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")

    print('Done.')

if __name__ == '__main__':
    main()