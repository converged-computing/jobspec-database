"""
Example of running a single experiment of unet in the head and neck data.
The json config of the main model is 'examples/json/unet-sample-config.json'
All experiment outputs are stored in '../../hn_perf/logs'.
After running 3 epochs, the performance of the training process can be accessed
as log file and perforamance plot.
In addition, we can peek the result of 42 first images from prediction set.
"""

import customize_obj
# import h5py
# from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from deoxys.experiment import Experiment, ExperimentPipeline
# from deoxys.model.callbacks import PredictionCheckpoint
# from deoxys.utils import read_file
import argparse
import os
# import numpy as np
# from pathlib import Path
# from comet_ml import Experiment as CometEx


if __name__ == '__main__':
    os.environ['NUM_CPUS'] = '4'
    os.environ['RAY_ROOT'] = '../../ray'
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("log_folder")
    # parser.add_argument("--temp_folder", default='', type=str)
    # parser.add_argument("--analysis_folder",
    #                     default='', type=str)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--model_checkpoint_period", default=5, type=int)
    parser.add_argument("--prediction_checkpoint_period", default=5, type=int)
    parser.add_argument("--meta", default='patient_idx', type=str)
    parser.add_argument("--monitor", default='', type=str)
    parser.add_argument("--memory_limit", default=0, type=int)

    args, unknown = parser.parse_known_args()

    if args.memory_limit:
        # Restrict TensorFlow to only allocate X-GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(
                    memory_limit=1024 * args.memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    base_folder = os.environ.get('base_folder', 'D:/cn_perf_{}/{}')
    log_folder = base_folder.format('local', args.log_folder)
    temp_folder = base_folder.format('temp', args.log_folder)

    if 'patch' in args.log_folder:
        analysis_folder = base_folder.format('analysis', args.log_folder)
        os.environ['ITER_PER_EPOCH'] = '100'
    else:
        analysis_folder = ''

    if '2d' in args.log_folder:
        meta = args.meta
    else:
        meta = args.meta.split(',')[0]

    print('training from configuration', args.config_file,
          'and saving log files to', log_folder)
    print('Unprocesssed prediciton are saved to', temp_folder)
    if analysis_folder:
        print('Intermediate processed files for merging patches are saved to',
              analysis_folder)

    exp = ExperimentPipeline(
        log_base_path=log_folder,
        temp_base_path=temp_folder
    ).from_full_config(
        args.config_file
    ).run_experiment(
        train_history_log=True,
        model_checkpoint_period=args.model_checkpoint_period,
        prediction_checkpoint_period=args.prediction_checkpoint_period,
        epochs=args.epochs,
    ).apply_post_processors(
        recipe='patch',
        analysis_base_path=analysis_folder,
        map_meta_data=meta,
    ).plot_performance(
    ).load_best_model(monitor=args.monitor)
    
    if analysis_folder:
        exp.plot_prediction(best_num=2, worst_num=0)
    
    # run test based on best model
    exp.run_test().apply_post_processors(
        recipe='patch',
        analysis_base_path=analysis_folder,
        map_meta_data=meta,
        run_test=True
    ).plot_3d_test_images(best_num=2, worst_num=0)
