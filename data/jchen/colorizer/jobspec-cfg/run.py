"""
Colorizer run.py
Jiahua Chen, Kathy Li, Sreshtaa Rajesh, Kiara Vong

python run.py --help
for usage. Trains the model and saves. 
"""
import os
import argparse
import tensorflow as tf

from tensorflow import keras

from data import Dataset
import hyperparameters as hp

from model import build_unet_model

from tensorboard_utils import CustomModelSaver

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_args():
    """Perform command-line argument parsing."""

    parser = argparse.ArgumentParser(
        description="Colorizing images!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="." + os.sep + "data" + os.sep,
        help="Location where the dataset is stored.",
    )
    parser.add_argument(
        "--load-checkpoint",
        default=None,
        help="""Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.""",
    )
    parser.add_argument(
        "--image",
        default="data/test/val_256/Places365_val_00000013.jpg",
        help="""Name of an image in the dataset to graph.""",
    )

    return parser.parse_args()


def train(model, datasets, logs_path="logs/", checkpoint_path="checkpoint/"):
    callback_list = [
        keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq="batch",
            profile_batch=0,
        ),
        CustomModelSaver(checkpoint_path, hp.max_num_weights),
    ]

    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        steps_per_epoch=hp.steps_per_epoch,
        validation_steps=hp.validation_steps,
        callbacks=callback_list,
    )


def test(model, test_data):
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    dataset = Dataset(ARGS.data)
    model = build_unet_model((hp.img_size, hp.img_size, 3))
    model(keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    print(model.summary())
    if ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.learning_rate),
        loss="mean_squared_error",
        metrics=[keras.metrics.MeanSquaredError()],
    )
    train(model, dataset)


ARGS = parse_args()
if __name__ == "__main__":
    main()
