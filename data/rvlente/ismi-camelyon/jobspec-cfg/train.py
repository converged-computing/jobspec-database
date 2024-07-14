import tensorflow_datasets as tfds
from tensorflow import keras
import tensorflow as tf
import models
import argparse
import os
import json

import albumentations as A

AUTOTUNE = tf.data.AUTOTUNE
CHECKPOINT_FILE = 'ckpt-best.hdf5'


def main(model_id, da, no_gpu):
    if no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print(f'Using DA level {da}')

    # Create model
    model, preprocess = models.create(model_id)
    model.summary()

    # Setup data augmentation.
    augmentations = [
    ]

    # Rotational
    if da > 0:
        augmentations.extend([
            A.Flip(),
            A.Rotate(180),
        ])

    # Color
    if da > 1:
        augmentations.extend([
            A.HueSaturationValue(),
            A.RandomBrightnessContrast(),
        ])

    # Distortion
    if da > 2:
        augmentations.extend([
            A.GridDistortion(4),
            A.Blur(3),
        ])

    augmentations = A.Compose(augmentations)

    def aug_fn(image):
        return augmentations(image=image)['image']

    # Load datasets
    pcam = tfds.load('patch_camelyon')

    def convert_sample(sample):
        image, label = sample['image'], sample['label']
        if da > 0:
            image = tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.uint8)
        image = preprocess(image)
        label = tf.one_hot(label, 2, dtype=tf.float32)
        return image, label

    train_pipeline = pcam['train'].map(convert_sample, num_parallel_calls=AUTOTUNE) \
        .shuffle(1024) \
        .repeat() \
        .batch(64) \
        .prefetch(AUTOTUNE)

    valid_pipeline = pcam['validation'].map(convert_sample, num_parallel_calls=AUTOTUNE) \
        .shuffle(1024) \
        .repeat() \
        .batch(64) \
        .prefetch(AUTOTUNE)

    # Train
    history = model.fit(
        train_pipeline,
        validation_data=valid_pipeline,
        verbose=2,
        epochs=30,
        steps_per_epoch=4096,
        validation_steps=256,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir='logs'
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=CHECKPOINT_FILE,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=True,
            ),
        ]
    )

    with open('history.json', 'w') as f:
        json.dump(history.history, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id')
    parser.add_argument('--da', type=int, choices=[0, 1, 2, 3], default=0, const=1, nargs='?')
    parser.add_argument('--no_gpu', action='store_const', const=True, default=False)
    main(**vars(parser.parse_args()))
