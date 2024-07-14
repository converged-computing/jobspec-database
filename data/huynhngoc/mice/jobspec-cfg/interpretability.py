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
from deoxys.experiment import DefaultExperimentPipeline
# from deoxys.model.callbacks import PredictionCheckpoint
# from deoxys.utils import read_file
import argparse
# import os
from deoxys.utils import read_csv
import numpy as np
# from pathlib import Path
# from comet_ml import Experiment as CometEx
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
import h5py
import gc
import pandas as pd


class Matthews_corrcoef_scorer:
    def __call__(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)

    def _score_func(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)


class Matthews_corrcoef_scorer:
    def __call__(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)

    def _score_func(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)


try:
    metrics.SCORERS['mcc'] = Matthews_corrcoef_scorer()
except:
    pass
try:
    metrics._scorer._SCORERS['mcc'] = Matthews_corrcoef_scorer()
except:
    pass


def metric_avg_score(res_df, postprocessor):
    res_df['avg_score'] = res_df[['AUC', 'roc_auc', 'f1', 'f1_0',
                                  'BinaryAccuracy', 'mcc']].mean(axis=1)

    return res_df


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument("log_folder")
    parser.add_argument("--temp_folder", default='', type=str)
    parser.add_argument("--model_checkpoint_period", default=1, type=int)
    parser.add_argument("--prediction_checkpoint_period", default=1, type=int)
    parser.add_argument("--meta", default='patient_idx', type=str)
    parser.add_argument(
        "--monitor", default='avg_score', type=str)
    parser.add_argument(
        "--monitor_mode", default='max', type=str)
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

    if '2d' in args.log_folder:
        meta = args.meta
    else:
        meta = args.meta.split(',')[0]

    print('external config from', args.dataset_file,
          'and explaining on models in', args.log_folder)
    print('Unprocesssed prediction are saved to', args.temp_folder)

    def binarize(targets, predictions):
        return targets, (predictions > 0.5).astype(targets.dtype)

    def flip(targets, predictions):
        return 1 - targets, 1 - (predictions > 0.5).astype(targets.dtype)

    exp = DefaultExperimentPipeline(
        log_base_path=args.log_folder,
        temp_base_path=args.temp_folder
    ).load_best_model(
        monitor=args.monitor,
        use_raw_log=False,
        mode=args.monitor_mode,
        custom_modifier_fn=metric_avg_score
    )

    seed = 1
    model = exp.model.model
    dr = exp.model.data_reader

    test_gen = dr.test_generator
    steps_per_epoch = test_gen.total_batch
    batch_size = test_gen.batch_size
    # pids
    pids = []
    with h5py.File(exp.post_processors.dataset_filename) as f:
        for fold in test_gen.folds:
            pids.append(f[fold][meta][:])
    pids = np.concatenate(pids)

    with h5py.File(args.log_folder + f'/test_vargrad_02.h5', 'w') as f:
        print('created file', args.log_folder + f'/test_vargrad_02.h5')
        f.create_dataset(meta, data=pids)
        f.create_dataset('vargrad', shape=(len(pids), 256, 256))
    data_gen = test_gen.generate()
    i = 0
    sub_idx = 0
    mc_preds = []
    tta_preds = []
    for x, _ in data_gen:
        print('MC results ....')
        tf.random.set_seed(seed)
        mc_pred = model(x).numpy().flatten()
        mc_preds.append(mc_pred)
        print(f'Batch {i}/{steps_per_epoch}')
        np_random_gen = np.random.default_rng(1123)
        new_shape = list(x.shape) + [40]
        var_grad = np.zeros(new_shape)
        tta_pred = np.zeros((x.shape[0], 40))
        for trial in range(40):
            print(f'Trial {trial+1}/40')
            noise = np_random_gen.normal(
                loc=0.0, scale=.02, size=x.shape[:-1]) * 255
            x_noised = x + np.stack([noise]*3, axis=-1)
            x_noised = tf.Variable(x_noised)
            tf.random.set_seed(seed)
            with tf.GradientTape() as tape:
                tape.watch(x_noised)
                pred = model(x_noised)

            grads = tape.gradient(pred, x_noised).numpy()
            var_grad[..., trial] = grads

            tta_pred[..., trial] = pred.numpy().flatten()

        tta_preds.append(tta_pred)
        final_var_grad = (var_grad.std(axis=-1)**2).mean(axis=-1)
        with h5py.File(args.log_folder + f'/test_vargrad_02.h5', 'a') as f:
            f['vargrad'][sub_idx:sub_idx + len(x)] = final_var_grad
        sub_idx += x.shape[0]
        i += 1
        gc.collect()
        if i == steps_per_epoch:
            break

    df = pd.DataFrame({'pid': pids, 'predicted': np.concatenate(mc_preds)})
    tta_preds = np.concatenate(tta_preds)
    for trial in range(40):
        df[f'tta_pred_{trial}'] = tta_preds[..., trial]

    df.to_csv(
        args.log_folder + f'/tta_predicted_02.csv', index=False)
