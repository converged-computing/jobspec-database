import numpy as np
import h5py
import argparse
from tqdm import tqdm
import csv
import tensorflow as tf
from tensorflow import keras
import json

import models

CHECKPOINT_FILE = 'ckpt-best.hdf5'


def main(out_file, model_id, x_test_path, y_test_path, tta):
    # Load test data.
    x_test = h5py.File(x_test_path)['x'][:100]
    y_test = h5py.File(y_test_path)['y'][:100]
    y_test = tf.squeeze(y_test)

    # Load model.
    model, preprocess = models.create(model_id)

    # Perform inference.
    model.load_weights(CHECKPOINT_FILE)
    preds = []

    for x in tqdm(x_test):
        x = preprocess(x)
        inputs = augment(x) if tta else np.expand_dims(x, axis=0)
        ys = model.predict(inputs)
        preds.append(ys.mean(axis=0)[1])

    # Report metrics.
    acc = keras.metrics.BinaryAccuracy()
    acc.update_state(y_test, preds)
    test_acc = float(acc.result().numpy())

    auc = keras.metrics.AUC()
    auc.update_state(y_test, preds)
    test_auc = float(auc.result().numpy())
    print(f'test accuracy, test auc: {test_acc}, {test_auc}')

    with open('eval.json', 'w') as f:
        json.dump({'accuracy': test_acc, 'auc': test_auc}, f)

    # Create submission file.
    with open(out_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('case', 'prediction'))
        writer.writerows(([i, f'{y:.6f}'] for i, y in enumerate(preds)))


def augment(x):
    augmentations = [
        tf.image.flip_left_right,        # horizontal flip
        tf.image.flip_up_down,           # vertical flip
        lambda x: tf.image.rot90(x, 1),  # 90 degrees
        lambda x: tf.image.rot90(x, 2),  # 180 degrees
        lambda x: tf.image.rot90(x, 3),  # 270 degrees
    ]
    return np.array([x, *[aug(x) for aug in augmentations]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_file')
    parser.add_argument('model_id')
    parser.add_argument('x_test_path')
    parser.add_argument('y_test_path')
    parser.add_argument('--tta', action='store_const', const=True, default=False)
    main(**vars(parser.parse_args()))
