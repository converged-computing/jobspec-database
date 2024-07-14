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
# import numpy as np
# from pathlib import Path
# from comet_ml import Experiment as CometEx
import numpy as np
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from lifelines.utils import concordance_index
from sklearn.metrics import roc_curve, auc, roc_auc_score


class Matthews_corrcoef_scorer:
    def __call__(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)

    def _score_func(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)


class CI_scorer:
    def __call__(self, *args, **kwargs):
        ci, *others = concordance_index_censored(
            args[0][..., 0] > 0, args[0][..., 1], args[1][..., 1].flatten(), **kwargs)
        return ci

    def _score_func(self, *args, **kwargs):
        ci, *others = concordance_index_censored(
            args[0][..., 0] > 0, args[0][..., 1], args[1][..., 0].flatten(), **kwargs)
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
        predicted_score = np.cumprod(y_pred[:, 0: np.where(
            breaks >= self.num_year*12)[0][0]], axis=1)[:, -1]
        return concordance_index(time, predicted_score, event)

    def _score_func(self, y_true, y_pred, **kwargs):
        # ci, *others = concordance_index_censored(args[0][..., 0] > 0, args[0][..., 1], args[1][..., 0].flatten(), **kwargs)
        event = y_true[:, -2]
        time = y_true[:, -1]
        no_time_interval = y_pred.shape[-1]
        breaks = np.arange(0, 61, 60//(no_time_interval))
        predicted_score = np.cumprod(y_pred[:, 0: np.where(
            breaks >= self.num_year*12)[0][0]], axis=1)[:, -1]
        return concordance_index(time, predicted_score, event)


class AUC_scorer:
    """
    AUC score on actual survival and predicted probability for each time interval
    """

    def __call__(self, y_true, y_pred, **kwargs):
        # first ten items of y_true: 1 if individual survived that interval, 0 if not.
        true = y_true[:, :10]
        return roc_auc_score(true, y_pred)

    def _score_func(self, y_true, y_pred, **kwargs):
        true = y_true[:, :10]
        score = roc_auc_score(true, y_pred)
        return roc_auc_score(true, y_pred)


class IBS_scorer_old:
    """
    Integrated Brier Score on actual survival and predicted probability over all time intervals
    """

    def __call__(self, y_true, y_pred, **kwargs):
        event = y_true[:, -2]
        time = y_true[:, -1]
        survival_train = np.array(list(zip(event, time)))
        dtype = [('event', bool), ('time', np.float64)]
        structured_survival_train = np.array(
            list(map(tuple, survival_train)), dtype=dtype)
        times = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
        score = integrated_brier_score(
            structured_survival_train, structured_survival_train, y_pred, times)
        return score

    def _score_func(self, y_true, y_pred, **kwargs):
        event = y_true[:, -2]
        time = y_true[:, -1]
        survival_train = np.array(list(zip(event, time)))
        dtype = [('event', bool), ('time', np.float64)]
        structured_survival_train = np.array(
            list(map(tuple, survival_train)), dtype=dtype)
        times = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
        score = integrated_brier_score(
            structured_survival_train, structured_survival_train, y_pred, times)
        return score


class IBS_scorer:
    """
    Integrated Brier Score on actual survival and predicted probability over all time intervals
    """

    def __call__(self, y_true, y_pred, **kwargs):
        event = y_true[:, -2]
        time = y_true[:, -1]
        survival_train = np.array(list(zip(event, time)))
        dtype = [('event', bool), ('time', np.float64)]
        structured_survival_train = np.array(
            list(map(tuple, survival_train)), dtype=dtype)
        times = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
        score = integrated_brier_score(
            structured_survival_train, structured_survival_train, np.cumprod(y_pred, axis=-1), times)
        return score

    def _score_func(self, y_true, y_pred, **kwargs):
        event = y_true[:, -2]
        time = y_true[:, -1]
        survival_train = np.array(list(zip(event, time)))
        dtype = [('event', bool), ('time', np.float64)]
        structured_survival_train = np.array(
            list(map(tuple, survival_train)), dtype=dtype)
        times = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
        score = integrated_brier_score(
            structured_survival_train, structured_survival_train, np.cumprod(y_pred, axis=-1), times)
        return score


try:
    metrics.SCORERS['mcc'] = Matthews_corrcoef_scorer()
    metrics.SCORERS['CI'] = CI_scorer()
    metrics.SCORERS['HCI'] = HCI_scorer()
    metrics.SCORERS['HCI_1yr'] = HCI_scorer(num_year=1)
    metrics.SCORERS['AUC'] = AUC_scorer()
    metrics.SCORERS['IBS'] = IBS_scorer_old()
    metrics.SCORERS['IBS_fixed'] = IBS_scorer()
except:
    pass
try:
    metrics._scorer._SCORERS['mcc'] = Matthews_corrcoef_scorer()
    metrics._scorer._SCORERS['CI'] = CI_scorer()
    metrics._scorer._SCORERS['HCI'] = HCI_scorer()
    metrics._scorer._SCORERS['HCI_1yr'] = HCI_scorer(num_year=1)
    metrics._scorer._SCORERS['AUC'] = AUC_scorer()
    metrics._scorer._SCORERS['IBS'] = IBS_scorer_old()
    metrics._scorer._SCORERS['IBS_fixed'] = IBS_scorer()
except:
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("log_base")
    parser.add_argument("name_list")
    parser.add_argument("--merge_name", default='merge', type=str)
    parser.add_argument("--mode", default='ensemble', type=str)
    parser.add_argument("--meta", default='patient_idx', type=str)
    parser.add_argument(
        "--monitor", default='HCI_5yr', type=str)
    parser.add_argument(
        "--monitor_mode", default='max', type=str)

    args, unknown = parser.parse_known_args()

    log_path_list = [args.log_base +
                     name for name in args.name_list.split(',')]
    log_base_path = args.log_base + args.merge_name

    if args.mode == 'ensemble':
        print('Ensemble test results from this list', log_path_list)
    else:
        print('Concatenate test results from this list', log_path_list)

    print('Merged results are save to', log_base_path)

    def binarize(targets, predictions):
        return targets, (predictions > 0.5).astype(targets.dtype)

    def flip(targets, predictions):
        return 1 - targets, 1 - (predictions > 0.5).astype(targets.dtype)

    pp = customize_obj.EnsemblePostProcessor(
        log_base_path=log_base_path,
        log_path_list=log_path_list,
        map_meta_data=args.meta.split(',')
    )

    if args.mode == 'ensemble':
        pp.ensemble_results()
    else:
        pp.concat_results()

    pp.calculate_metrics(
        metrics=['HCI', 'AUC', 'IBS', 'IBS_fixed'],
        metrics_sources=['sklearn', 'sklearn', 'sklearn', 'sklearn'],
        process_functions=[None, None, None, None],
        metrics_kwargs=[{'metric_name': 'HCI_5yr'}, {}, {}, {}]
    )
