"""
Example of running a single experiment of unet in the head and neck data.
The json config of the main model is 'examples/json/unet-sample-config.json'
All experiment outputs are stored in '../../hn_perf/logs'.
After running 3 epochs, the performance of the training process can be accessed
as log file and perforamance plot.
In addition, we can peek the result of 42 first images from prediction set.
"""

from deoxys.experiment import DefaultExperimentPipeline
# from deoxys.utils import read_file
import argparse
import os
import shutil
from deoxys.utils import read_csv
import numpy as np
# from pathlib import Path
# from comet_ml import Experiment as CometEx
import tensorflow as tf
import customize_obj

from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from lifelines.utils import concordance_index



class Matthews_corrcoef_scorer:
    def __call__(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)

    def _score_func(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)

class CI_scorer:
    def __call__(self, *args, **kwargs):
        ci, *others = concordance_index_censored(args[0][..., 0] > 0, args[0][..., 1], args[1][..., 0].flatten(), **kwargs)
        return ci

    def _score_func(self, *args, **kwargs):
        ci, *others = concordance_index_censored(args[0][..., 0] > 0, args[0][..., 1], args[1][..., 0].flatten(), **kwargs)
        return ci

class HCI_scorer:
    def __init__(self, num_year=5):
        self.num_year = num_year

    def __call__(self, y_true, y_pred, **kwargs):
        # ci, *others = concordance_index_censored(args[0][..., 0] > 0, args[0][..., 1], args[1][..., 1].flatten(), **kwargs)
        event = y_true[:, -2]
        time = y_true[:, -1]
        no_time_interval = y_pred.shape[-1]
        breaks = np.arange(0, 61, 60//(no_time_interval))
        predicted_score = np.cumprod(y_pred[:,0: np.where(breaks>=self.num_year*12)[0][0]], axis=1)[:,-1]
        return concordance_index(time, predicted_score, event)

    def _score_func(self, y_true, y_pred, **kwargs):
        #ci, *others = concordance_index_censored(args[0][..., 0] > 0, args[0][..., 1], args[1][..., 0].flatten(), **kwargs)
        event = y_true[:, -2]
        time = y_true[:, -1]
        no_time_interval = y_pred.shape[-1]
        breaks = np.arange(0, 61, 60//(no_time_interval))
        predicted_score = np.cumprod(y_pred[:,0: np.where(breaks>=self.num_year*12)[0][0]], axis=1)[:,-1]
        return concordance_index(time, predicted_score, event)

class AUC_scorer:
    """
    AUC score on actual survival and predicted probability for each time interval
    """
    def __call__(self, y_true, y_pred, **kwargs):
        true = y_true[:, :10]  # first ten items of y_true: 1 if individual survived that interval, 0 if not.
        return roc_auc_score(true, y_pred)

    def _score_func(self, y_true, y_pred, **kwargs):
        true = y_true[:, :10]
        return roc_auc_score(true, y_pred)

class IBS_scorer:
    """
    Integrated Brier Score on actual survival and predicted probability over all time intervals
    """
    def __call__(self, y_true, y_pred, **kwargs):
        event = y_true[:, -2]
        time = y_true[:, -1]
        survival_train = np.array(list(zip(event, time)))
        dtype = [('event', bool), ('time', np.float64)]
        structured_survival_train = np.array(list(map(tuple, survival_train)), dtype=dtype)
        times = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
        score = integrated_brier_score(structured_survival_train, structured_survival_train, y_pred, times)
        return score

    def _score_func(self, y_true, y_pred, **kwargs):
        event = y_true[:, -2]
        time = y_true[:, -1]
        survival_train = np.array(list(zip(event, time)))
        dtype = [('event', bool), ('time', np.float64)]
        structured_survival_train = np.array(list(map(tuple, survival_train)), dtype=dtype)
        times = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
        score = integrated_brier_score(structured_survival_train, structured_survival_train, y_pred, times)
        return score


try:
    metrics.SCORERS['mcc'] = Matthews_corrcoef_scorer()
    metrics.SCORERS['CI'] = CI_scorer()
    metrics.SCORERS['HCI'] = HCI_scorer()
    metrics.SCORERS['AUC'] = AUC_scorer()
    metrics.SCORERS['IBS'] = IBS_scorer()
except:
    pass
try:
    metrics._scorer._SCORERS['mcc'] = Matthews_corrcoef_scorer()
    metrics._scorer._SCORERS['CI'] = CI_scorer()
    metrics._scorer._SCORERS['HCI'] = HCI_scorer()
    metrics._scorer._SCORERS['AUC'] = AUC_scorer()
    metrics._scorer._SCORERS['IBS'] = IBS_scorer()
except:
    pass


def metric_avg_score(res_df, postprocessor):
    auc = res_df['AUC']
    mcc = res_df['mcc'] / 2 + 0.5
    f1 = res_df['f1']
    f0 = res_df['f1_0']

    # get f1 score in train data
    epochs = res_df['epochs']
    train_df = read_csv(
        postprocessor.log_base_path + '/logs.csv')
    train_df['real_epoch'] = train_df['epoch'] + 1
    train_f1 = train_df[train_df.real_epoch.isin(epochs)]['BinaryFbeta'].values
    train_f1 = 2 * np.sqrt(train_f1) / 3

    res_df['avg_score'] = (auc + mcc + f1 + 0.75*f0 + 0.75*train_f1) / 4.5

    return res_df

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument("log_folder")
    parser.add_argument("--best_epoch", default=0, type=int)
    parser.add_argument("--temp_folder", default='', type=str)
    parser.add_argument("--analysis_folder",
                        default='', type=str)
    parser.add_argument("--meta", default='patient_idx', type=str)
    parser.add_argument(
        "--monitor", default='HCI_5yr', type=str)
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

    # if '2d' in args.log_folder:
    #     meta = args.meta
    # else:
    #     meta = args.meta.split(',')[0]
    meta = args.meta

    def binarize(targets, predictions):
        return targets, (predictions > 0.5).astype(targets.dtype)

    def flip(targets, predictions):
        return 1 - targets, 1 - (predictions > 0.5).astype(targets.dtype)

    # copy to another location
    log_folder = args.log_folder + '_' + args.dataset_file[:-5].split('/')[-1]
    if not os.path.exists(log_folder):
        shutil.copytree(args.log_folder, log_folder)

    if not os.path.exists(log_folder + '/model'):
        shutil.copytree(args.log_folder + '/model', log_folder + '/model')

    ex = DefaultExperimentPipeline(
        log_base_path=log_folder,
        temp_base_path=args.temp_folder + '_' +
        args.dataset_file[:-5].split('/')[-1]
    )
    if args.best_epoch == 0:
        try:
            ex = ex.load_best_model(
                monitor=args.monitor,
                use_raw_log=False,
                mode=args.monitor_mode,
                # custom_modifier_fn=metric_avg_score
            )
        except Exception as e:
            print("Error while loading best model", e)
            print(e)
    else:
        print(f'Loading model from epoch {args.best_epoch}')
        ex.from_file(args.log_folder +
                     f'/model/model.{args.best_epoch:03d}.h5')
    ex.run_external(
        args.dataset_file
    ).apply_post_processors(
        map_meta_data=meta, run_test=True,
        metrics=['HCI', 'AUC', 'IBS'],
        metrics_sources=['sklearn', 'sklearn', 'sklearn'],
        process_functions=[None, None, None],
        metrics_kwargs=[{'metric_name': 'HCI_5yr'}, {}, {}]
    )

    # shutil.rmtree(log_folder + '/model')
