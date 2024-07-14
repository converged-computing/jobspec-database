"""
Use the experiment module in the deoxys package to train different CNN models
configured in ../config/file_name.json
"""

from deoxys.experiment import Experiment
from deoxys.utils import read_file
import argparse
import tensorflow as tf


if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("log_folder")
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--model_checkpoint_period", default=25, type=int)
    parser.add_argument("--prediction_checkpoint_period", default=25, type=int)

    args = parser.parse_args()

    config = read_file(args.config_file)

    exp = Experiment(
        log_base_path=args.log_folder
    ).from_full_config(
        config
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=args.model_checkpoint_period,
        prediction_checkpoint_period=args.prediction_checkpoint_period,
        epochs=args.epochs,
    )
