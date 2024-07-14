"""
Linear classification evaluation of pretrained features. Needs the following packages:
- pytorch-lightning
- scikit-learn
- torch
- pandas
- (optional) scikit-learn-intelex
- (optional) imbalanced-learn
- (optional) iterative-stratification
- (optional) lightning-bolts

Eg of command to run:
-  For hyper parameter tuning and balancing the losses:
`python eval_pretrained_features.py --feature-path <feature_dir> --out-path <out_dir> --is-balance-loss --is-validation`
-  Standard (eg Imagenet) with standard eval head:
`python eval_pretrained_features.py --feature-path <feature_dir> --out-path <out_dir>`
-  Sklearn head (slow if multi label++):
-  Sklearn head (slow if multi label++):
`python eval_pretrained_features.py --feature-path <feature_dir> --out-path <out_dir> --is-sklearn`


To load back the results in a dataframe for plotting use:
`pd.read_csv("<out_path>/all_metrics.csv",index_col=[0,1,2], header=[0,1])`

The only function to change for different projects should be:
- preprocess_labels
- path_to_model
- load_train_val_test_features
"""


try:
    from sklearnex import patch_sklearn

    patch_sklearn(["LogisticRegression"])
except:
    # tries to speedup sklearn if possible (has to be before import sklearn)
    pass

import argparse
import logging
import os
import sys
from copy import deepcopy
from itertools import chain
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import torch.nn as nn
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    log_loss,
    make_scorer,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import (
    ParameterSampler,
    PredefinedSplit,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.utils import _safe_indexing, indexable
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
import time
from datetime import timedelta
from vissl.utils.extract_features_utils import ExtractedFeaturesLoader

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ImportError:
    # only needed if stratification of multi label data
    pass

try:
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    # only needed if you want balance train or val
    pass

try:
    from pytorch_lightning.loggers import WandbLogger
except:
    pass

try:
    from pl_bolts.optimizers.lars import LARS
except:
    pass

RAND_ID = random.randint(0,100000)
METRICS_FILENAME = "all_metrics.csv"
REPORT_FILENAME = "{split}_clf_report.csv"
PRED_FILENAME = "{split}_predictions.npz"

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s", datefmt="%d-%b %H:%M"
)


def main(cfg):
    logging.info(f"RAND_ID {RAND_ID}.")
    pl.seed_everything(0)  # shouldn't be needed
    metrics_base, dir_base = dict(), cfg.out_path
    for path in list(cfg.feature_path.glob(cfg.feature_pattern)):
        metrics_model, dir_model = init_level_(
            "model", path_to_model(path), metrics_base, dir_base
        )

        logging.info(f"Evaluating pretrained features from {path}.")
        Z_train, Y_train, Z_val, Y_val, Z_test, Y_test = load_train_val_test_features(
            path
        )

        # optional label preprocessing
        Y_train, Y_val, Y_test = preprocess_labels(path, Y_train, Y_val, Y_test)

        # optional validation split
        if cfg.is_validation:
            Z_train, Z_val, Y_train, Y_val = get_val_features(
                Z_train, Y_train, Z_val, Y_val
            )
        else:
            logging.info(f"Skipping validation / hparam tuning.")
            Z_val, Y_val = None, None


        assert type_of_target(Y_train) == type_of_target(Y_test)
        assert type_of_target(Y_train) != "multiclass-multioutput"
        cfg.type_of_target = type_of_target(Y_train)
        logging.info(f"This is a {cfg.type_of_target} task.")

        for train_size, n_epoch in zip(cfg.train_sizes, cfg.n_epochs):
            metrics_size, dir_size = init_level_(
                "train_size", train_size, metrics_model, dir_model
            )

            logging.info(f"Evaluating train_size={train_size} with n_epoch={n_epoch}.")
            cfg.curr_train_size = train_size
            cfg.curr_n_epoch = n_epoch

            for seed in range(cfg.n_runs):
                metrics_seed, dir_seed = init_level_(
                    "seed", seed, metrics_size, dir_size
                )
                dir_seed.mkdir(parents=True, exist_ok=True)

                try:
                    metrics_path = dir_seed / METRICS_FILENAME
                    metrics_seed["metrics"] = pd.read_csv(
                        metrics_path, index_col="split"
                    )
                    logging.info(
                        f"Skipping evaluation of seed {seed} as metrics were found at {metrics_path}."
                    )
                    continue
                except FileNotFoundError:
                    pass

                logging.info(f"Evaluating seed {seed} out of {cfg.n_runs}.")
                pl.seed_everything(seed)
                train_dataset = get_dataset(
                    Z_train,
                    Y_train,
                    is_balance_subset=cfg.is_balance_subset,
                    balance_data_mode=cfg.balance_data_mode,
                    size=train_size,
                    seed=seed,
                )
                val_dataset = get_dataset(Z_val, Y_val)
                test_dataset = get_dataset(Z_test, Y_test)

                if cfg.is_monitor_test:
                    val_dataset = test_dataset

                logging.info(f"Training + tuning the linear probe.")
                start = time.time()
                trainer = train(train_dataset, val_dataset, cfg, seed)
                logging.info(f"Done training + tuning. Time: {str(timedelta(seconds=time.time() - start)).split('.')[0]}.")

                eval_datasets = dict(test=test_dataset, train=train_dataset)
                for name, dataset in eval_datasets.items():
                    metrics_split, _ = init_level_(
                        "split", name, metrics_seed, dir_size
                    )
                    report_path = dir_seed / REPORT_FILENAME.format(split=name)
                    predictions_path = dir_seed / PRED_FILENAME.format(split=name)

                    logging.info(
                        f"Predicting {name}." # and saving Y,Yhat,(Yhat_score) to {predictions_path}."
                    )
                    Yhat, Yhat_score, Y = predict(trainer, dataset, cfg.is_sklearn)
                    to_save = dict(Yhat=Yhat, Y=Y)
                    if name != "train" and Yhat_score is not None:
                        # don't save proba for train because can be large and not useful (?)
                        to_save["Yhat_score"] = Yhat_score
                    #np.savez(predictions_path, **to_save) # don't save (memory ++)

                    logging.info(
                        f"Evaluating {name} and saving report to {report_path}."
                    )
                    metrics_split["metrics"], clf_report = evaluate(Yhat, Yhat_score, Y)
                    clf_report.to_csv(report_path)

                save_and_aggregate_metrics_("split", metrics_seed, dir_seed)
            save_and_aggregate_metrics_(
                "seed", metrics_size, dir_size, is_avg_over=True
            )  # avg over seeds
        save_and_aggregate_metrics_("train_size", metrics_model, dir_model)
    save_and_aggregate_metrics_("model", metrics_base, dir_base)


def init_level_(level, value, prev_dict, prev_dir):
    """Initialize metrics and save dir for hierarchical level (model, train size, seed)..."""
    prev_dict[value] = dict()
    new_dict = prev_dict[value]
    new_dir = prev_dir / f"{level}_{value}"
    return new_dict, new_dir


def save_and_aggregate_metrics_(level, prev_dict, prev_dir, is_avg_over=False):
    """Aggregate all the metrics from the current loop (either concat or avg)."""
    metrics = {k: v["metrics"] for k, v in prev_dict.items()}
    if len(metrics) > 1:
        metrics_path = prev_dir / METRICS_FILENAME
        logging.info(f"Saving aggregated metrics over {level} to {metrics_path}.")
        if isinstance(list(metrics.values())[0], pd.Series):
            agg = pd.DataFrame(metrics).T
        else:  # dataframes
            agg = pd.concat(metrics, axis=0)
        old_idx = agg.index.names[1:]
        agg.index.names = [level] + old_idx
        if is_avg_over:
            agg = agg.groupby(old_idx).agg(["mean", "sem"], axis=1)
        agg.to_csv(metrics_path)
    else:
        agg = None
    prev_dict["metrics"] = agg

##### DATA #####
def get_val_features(Z_train, Y_train, Z_val, Y_val):
    """Split the train and val if necessary."""

    if Z_val is None:
        # no existing existing split
        Z_train, Z_val, Y_train, Y_val = multilabel_train_test_split(
            Z_train, Y_train, stratify=Y_train, test_size=0.1, random_state=0
        )

    return Z_train, Z_val, Y_train, Y_val


def get_dataset(
    Z, Y, is_balance_subset=False, balance_data_mode=None, size=-1, seed=0
):
    """Return SklearnDataset with desired size and optional balancing."""
    if Z is None and Y is None:
        return None

    if size != -1:
        logging.info(f"Subsetting {size} examples.")

        if is_balance_subset:
            assert "imblearn" in sys.modules, "pip install -U imbalanced-learn"
            logging.info(f"Using balanced subset instead of stratified.")
            Z, Y = RandomUnderSampler(random_state=seed).fit_resample(Z, Y)
            assert size <= len(
                Z
            ), "If balancing need to have selected size smaller than under sampled."

        _, Z, _, Y = multilabel_train_test_split(
            Z, Y, stratify=Y, test_size=size, random_state=seed
        )

    if balance_data_mode is not None:
        assert "imblearn" in sys.modules, "pip install -U imbalanced-learn"
        if balance_data_mode == "undersample":
            Z, Y = RandomUnderSampler(random_state=seed).fit_resample(Z, Y)
        elif balance_data_mode == "oversample":
            Z, Y = RandomOverSampler(random_state=seed).fit_resample(Z, Y)
        else:
            raise ValueError(f"Unknown balance_data_mode={balance_data_mode}.")

    return SklearnDataset(Z, Y)


class SklearnDataset(Dataset):
    def __init__(self, Z, Y):
        super().__init__()
        self.Z = Z
        self.Y = Y

        tgt_type = type_of_target(self.Y)
        self.is_multilabel_tgt = tgt_type in ["multiclass-multioutput", "multilabel-indicator"]
        self.is_binary_tgt = tgt_type in ["binary", "multilabel-indicator"]
        self.is_multiclass_tgt = "multiclass" in tgt_type

    def __len__(self):
        return len(self.Z)

    def __getitem__(self, idx):
        Z = self.Z[idx].astype(np.float32)
        if self.is_binary_tgt:
            # BCE requires float as input
            Y = self.Y[idx].astype(np.float32)
        else:
            Y = self.Y[idx].astype(int)
        return Z, Y


def multilabel_train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
):
    """
    Train test split that uses improved algorithm for multi label from:
    'Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of Multi-Label Data'.

    The issue with sklearn's `train_test_split` is that it treats every combination of labels as single
    => often error if a combination is only seen once. Here takes into account also individual labels
    if needed.
    """
    if stratify is None or type_of_target(stratify) != "multilabel-indicator":
        return train_test_split(
            *arrays,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            stratify=stratify,
            shuffle=shuffle,
        )

    assert shuffle, "Stratified train/test split is not implemented for shuffle=False"
    assert "iterstrat" in sys.modules, "pip install iterative-stratification"

    arrays = indexable(*arrays)
    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )
    cv = MultilabelStratifiedShuffleSplit(
        test_size=n_test, train_size=n_train, random_state=random_state
    )
    train, test = next(cv.split(X=arrays[0], y=stratify))

    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )


###############

##### Training #####


def train(train_dataset, val_dataset, cfg, seed):
    """Train linear probe."""
    is_balance_val = cfg.is_balance_loss or cfg.balance_data_mode is not None

    if cfg.is_sklearn:
        clf = get_sklearn_clf(cfg, seed)

        if cfg.is_validation:
            # when using MultiOutputClassifier needs to change params to `estimator__*`
            prfx = "estimator__" if train_dataset.is_multilabel_tgt else ""

            # when pipeline (feature scaling) needs to change params to `clf__*`
            prfx += "clf__" if cfg.is_scale_features else ""

            param_space = dict()
            for param in cfg.sk_validate_param:
                if param == "C":
                    param_space[f"{prfx}C"] = loguniform(1e-4, 100)
                elif param == "penalty":
                    param_space[f"{prfx}penalty"] = ["l1", "l2"]

            Z = np.concatenate((train_dataset.Z, val_dataset.Z))
            Y = np.concatenate((train_dataset.Y, val_dataset.Y))

            # could replace that by LogisticRegressionCV for logistic regression
            clf = RandomizedSearchCV(
                clf,
                param_space,
                scoring=make_scorer(accuracy, is_balance=is_balance_val),
                # MultiOutputClassifier already uses parallel
                n_jobs=None,# if train_dataset.is_multilabel_tgt else -1,
                cv=PredefinedSplit(
                    [-1] * len(train_dataset.Z) + [0] * len(val_dataset.Z)
                ),
                n_iter=cfg.n_hyper_param,
                verbose=0 if cfg.is_no_progress_bar else 1,
                random_state=seed,
            )
        else:
            Z = train_dataset.Z
            Y = train_dataset.Y

        logging.info(f"Fitting sklearn {clf}.")
        clf = clf.fit(Z, Y)

        return clf

    else:
        callbacks = []
        if not cfg.no_wandb:
            callbacks += [LearningRateMonitor()]

        if not cfg.is_no_progress_bar:
            callbacks += [TQDMProgressBar(refresh_rate=600)]


        pl.seed_everything(seed)
        trainer_kwargs = dict(
            max_epochs=cfg.curr_n_epoch,
            log_every_n_steps=600,
            gpus=cfg.n_gpus,
            precision=16,
            enable_progress_bar=not cfg.is_no_progress_bar,
            limit_val_batches=1.0 if cfg.is_monitor_test else 0,
            check_val_every_n_epoch=20,
            fast_dev_run=False,
            enable_checkpointing=False,
            callbacks=callbacks,
            logger=None if cfg.no_wandb else  WandbLogger(project='vissl',
                               entity='yanndubs',
                               config=vars(cfg),
                               id=str(cfg.out_path).split("/")[-1] + f"_{RAND_ID}",
                               group=str(cfg.out_path).split("/")[0])
        )

        if cfg.is_validation:
            param_space = dict()
            for param in cfg.torch_validate_param:
                if param == "lr":
                    param_space["lr"] = loguniform(5e-2, 1)
                elif param == "is_batchnorm":
                    param_space["is_batchnorm"] = [True, False]
                elif param == "batch_size":
                    param_space["batch_size"] = [int(2 ** i) for i in range(6, 10)]
                elif param == "weight_decay":
                    param_space["weight_decay"] = loguniform(1e-7, 1e-5)

            param_list = list(
                ParameterSampler(
                    param_space, n_iter=cfg.n_hyper_param, random_state=seed
                )
            )

            best_metric = 0
            for sampled_params in param_list:

                cfg_tuning = deepcopy(cfg)
                cfg_tuning.__dict__.update(**sampled_params)

                clf = Probe(train_dataset, cfg_tuning)
                trainer = pl.Trainer(**trainer_kwargs)
                trainer.fit(clf)

                Yhat_val, _, Y_val = predict(trainer, val_dataset, is_sklearn=False)
                curr_metric = accuracy(Y_val, Yhat_val, is_balance=cfg.is_balance_loss)
                logging.info(
                    f"Temporary validation metric: {curr_metric} for {sampled_params} on balance={is_balance_val}."
                )

                if curr_metric > best_metric:
                    best_params = sampled_params
                    best_metric = curr_metric
                    best_trainer = trainer

            logging.info(
                f"Selected parameters after validation: {best_params}, metric: {best_metric}."
            )
        else:

            clf = Probe(train_dataset, cfg, val_dataset=val_dataset)
            best_trainer = pl.Trainer(**trainer_kwargs)
            best_trainer.fit(clf)

        return best_trainer


###############

##### PREDICITING #####


def predict(trainer, dataset, is_sklearn):
    """Return predicted label, score (confidence or proba if available), true target."""
    if is_sklearn:
        clf = trainer
        Y = dataset.Y
        Yhat = clf.predict(dataset.Z)

        if hasattr(clf, "predict_proba"):
            Yhat_score = clf.predict_proba(dataset.Z)
            if dataset.is_binary_tgt:
                # squeeze probabilities if binary
                if isinstance(Yhat_score, list):
                    Yhat_score = np.concatenate([s[:, 1] for s in Yhat_score], axis=1)
                else:
                    Yhat_score = Yhat_score[:, 1]

        elif hasattr(clf, "decision_function"):
            Yhat_score = clf.decision_function(dataset.Z)
            if isinstance(Yhat_score, list):
                Yhat_score = np.concatenate(Yhat_score, axis=1)
        else:
            logging.info(f"Cannot compute scores / proba for {clf}. Skipping.")
            Yhat_score = None
            # eg multiOutputClassifier with linearSVC won't work for now
            # see : https://github.com/scikit-learn/scikit-learn/issues/21861

    else:
        clf = trainer.lightning_module
        predicted = trainer.predict(clf, dataloaders=clf.eval_dataloader(dataset))
        Yhat_score, Y = zip(*predicted)
        Y = np.concatenate(Y, axis=0)
        Yhat_score = np.concatenate(Yhat_score, axis=0)

        if dataset.is_binary_tgt:
            Yhat = (Yhat_score > 0.5).astype(int)
        elif dataset.is_multiclass_tgt:
            Yhat = Yhat_score.argmax(axis=1)

    return Yhat, Yhat_score, Y


#######################


##### EVALUATION ######
def evaluate(Yhat, Yhat_score, Y):
    """Compute many useful classification metrics."""

    # avoid slow computations if large (eg imagenet train)
    tgt_type = type_of_target(Y)
    is_many_labels = "multilabel" in tgt_type and Y.shape[1] > 100
    is_many_classes = "multiclass" in tgt_type and len(np.unique(Y)) > 100
    is_many_samples = len(Y) > 6e4  # max is imagenet val
    is_large = is_many_samples and (is_many_classes or is_many_labels)

    clf_report = pd.DataFrame(classification_report(Y, Yhat, output_dict=True, zero_division=0)).T

    metrics = dict()
    metrics["accuracy"] = accuracy(Y, Yhat, is_balance=False)
    metrics["balanced_accuracy"] = accuracy(Y, Yhat, is_balance=True)

    try:
        prfs = precision_recall_fscore_support(Y, Yhat, average="weighted", zero_division=0)
        for name, metric in zip(["precision", "recall", "f1", "support"], prfs):
            if metric is not None:
                # support will be none because average is weighted
                metrics[f"weighted_{name}"] = metric

        if Yhat_score is not None and not is_large:
            # all of this is skipped for imagenet train because slow + memory intensive

            if tgt_type == "multiclass":
                metrics["top5_accuracy"] = top_k_accuracy_score(Y, Yhat_score, k=5)

            if "multilabel" not in tgt_type:
                # could deal with multi label but annoying and not that useful
                metrics["log_loss"] = log_loss(Y, Yhat_score)

            metrics["auc"] = roc_auc_score(Y, Yhat_score, average="weighted", multi_class="ovr")

    except:
        logging.exception("Skipping secondary metrics which failed with error:")

    metrics = pd.Series(metrics)
    return metrics, clf_report


def mean(l):
    """Mean of a list."""
    return sum(l) / len(l)


def accuracy(Y_true, Y_pred, is_balance=False):
    """Computes the (balanced) accuracy."""
    if is_balance:
        if Y_true.ndim == 2:
            acc = mean(
                [
                    balanced_accuracy_score(Y_true[:, i], Y_pred[:, i])
                    for i in range(Y_true.shape[1])
                ]
            )
        else:
            acc = balanced_accuracy_score(Y_true, Y_pred)
    else:
        acc = accuracy_score(Y_true.flatten(), Y_pred.flatten())
    return acc


######################

##### SKLEARN SPECIFIC #####
def get_sklearn_clf(cfg, seed):
    """Return the correct sklearn classifier."""
    shared_kwargs = dict(
        C=cfg.C,
        class_weight="balanced" if cfg.is_balance_loss else None,
        random_state=seed,
        tol=1e-3,
    )

    is_multilabel_tgt = cfg.type_of_target in [
        "multiclass-multioutput",
        "multilabel-indicator",
    ]
    if cfg.is_svm:
        # primal should be quicker when more samples than features
        clf = LinearSVC(dual=False, **shared_kwargs)
    else:
        # don't use parallel if parallelize over hyperparameters or multi output already
        n_jobs = None if (is_multilabel_tgt) else -1 #cfg.is_validation or
        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            n_jobs=n_jobs,
            warm_start=True,
            **shared_kwargs,
        )

    if is_multilabel_tgt:
        # in case multi label then fit one clf on each
        clf = MultiOutputClassifier(clf, n_jobs=-1)

    if cfg.is_scale_features:
        clf = Pipeline([("scaler", MinMaxScaler()), ("clf", clf)])

    return clf


###############

##### TORCH SPECIFIC ######
class Probe(pl.LightningModule):
    """Linear or MLP probe."""

    def __init__(self, train_dataset, cfg, val_dataset=None):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        Y = self.train_dataset.Y

        if self.train_dataset.is_multiclass_tgt:
            out_size = len(np.unique(Y))
            if cfg.is_balance_loss:
                weight = torch.from_numpy(
                    compute_class_weight("balanced", classes=np.unique(Y), y=Y)
                ).float()
            else:
                weight = None
            self.criterion = nn.CrossEntropyLoss(weight=weight)

        elif self.train_dataset.is_binary_tgt:
            if Y.ndim == 1:
                Y = np.expand_dims(Y, 1)

            out_size = Y.shape[1]
            if cfg.is_balance_loss:
                n_pos = Y.sum(0)
                assert not (n_pos == 0).any()
                n_neg = Y.shape[0] - n_pos
                pos_weight = torch.from_numpy(n_neg / n_pos).float()
            else:
                pos_weight = None
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        in_size = self.train_dataset.Z.shape[1]

        if cfg.is_mlp:
            hidden_size = 2048
            self.probe = nn.Sequential(nn.Linear(in_size, hidden_size),
                                       nn.BatchNorm1d(hidden_size),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.BatchNorm1d(hidden_size),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(in_size, out_size),
            )
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm1d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

        elif cfg.is_mlpS:
            hidden_size = 2048
            self.probe = nn.Sequential(nn.Linear(in_size, hidden_size),
                                       nn.BatchNorm1d(hidden_size),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(in_size, out_size),
            )
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm1d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

        else:
            self.probe = nn.Linear(in_size, out_size)
            nn.init.trunc_normal_(self.probe.weight, std=0.02)
            nn.init.zeros_(self.probe.bias)


        if cfg.is_batchnorm:
            # normalize features before probe
            self.probe = nn.Sequential(
                nn.BatchNorm1d(in_size, affine=False), self.probe
            )

    @property
    def max_num_workers(self):
        try:
            max_num_workers = len(os.sched_getaffinity(0))
        except:
            max_num_workers = os.cpu_count()
        return max_num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.max_num_workers - 1,
            pin_memory=True,
        )

    def val_dataloader(self):
        return self.eval_dataloader(self.val_dataset)

    def eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.max_num_workers - 1,
            pin_memory=True,
        )

    def forward(self, z):
        logits = self.probe(z).squeeze()

        if self.train_dataset.is_multiclass_tgt:
            out = logits.softmax(-1)  # for probs
            # out = logits.argmax(dim=1) # for labels

        elif self.train_dataset.is_binary_tgt:
            out = logits.sigmoid()  # for probs
            # out = (logits > 0).int() # for labels

        return out

    def step(self, batch, mode):
        z, y = batch
        Y_logits = self.probe(z).squeeze()
        loss = self.criterion(Y_logits, y)

        if self.train_dataset.is_binary_tgt:
            Y_hat = (Y_logits > 0)

        elif self.train_dataset.is_multiclass_tgt:
            Y_hat = Y_logits.argmax(dim=-1)


        logs = dict()
        logs["acc"] = (Y_hat.float() == y).float().mean()
        logs["loss"] = loss
        self.log_dict({f"{mode}/{k}": v for k, v in logs.items()})

        return loss

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x).cpu().numpy(), y.cpu().numpy()

    def configure_optimizers(self):
        # standard linear lr scaling
        linear_lr = self.hparams.lr * self.hparams.batch_size / 256

        if self.hparams.is_lars:
            optimizer = LARS(
                self.probe.parameters(),
                lr=linear_lr,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9,
            )
        elif self.hparams.is_adamw:
            optimizer = torch.optim.AdamW(
                self.probe.parameters(),
                lr=linear_lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                self.probe.parameters(),
                lr=linear_lr,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9,
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.hparams.curr_n_epoch, eta_min=0
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


###############

### PROJECT / DATA / MODEL SPECIFIC ###
def load_train_val_test_features(path):
    """
    Load and return train, val, test **np array** of the pretrained features and targets for a path.
    If no validation return None for Z_val and Y_val. If using single label, that target array should be squeezed.
    """
    kwargs = dict(input_dir=path, layer="heads", flatten_features=False,)
    features = ExtractedFeaturesLoader.load_features(split="train", **kwargs)
    Z_train = features['features']
    Y_train = features['targets']

    features = ExtractedFeaturesLoader.load_features(split="test", **kwargs)
    Z_test = features['features']
    Y_test = features['targets']

    try:
        features = ExtractedFeaturesLoader.load_features(split="val", **kwargs)
        Z_val = features['features']
        Y_val = features['targets']
    except ValueError:
        Z_val, Y_val = None, None

    return (
        Z_train,
        Y_train,
        Z_val,
        Y_val,
        Z_test,
        Y_test,
    )

def preprocess_labels(path, Y_train, Y_val, Y_test):
    """Applies the desired label preprocessing."""
    if Y_val is not None:
        Y_val = Y_val.squeeze()
    return Y_train.squeeze(), Y_val, Y_test.squeeze()

def path_to_model(path):
    """Return model name from path."""
    epoch = str(path).split("phase")[-1]
    model = str(path).split("_dir/")[0].split("/")[-1]
    return f"{model}_epoch{epoch}"

################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Linear evaluation of models")
    general_args = parser.add_argument_group("general")
    train_args = parser.add_argument_group("model training")
    torch_args = parser.add_argument_group("torch specific")
    sk_args = parser.add_argument_group("sklearn specific")
    data_args = parser.add_argument_group("data")

    general_args.add_argument(
        "--feature-path", required=True, help="path to pretrained features"
    )
    general_args.add_argument(
        "--out-path", required=True, help="path to outputs and metrics"
    )
    general_args.add_argument(
        "--feature-pattern", default="model_*", help="glob pattern to find features."
    )
    general_args.add_argument(
        "--n-runs", default=1, type=int, help="number of evaluation to do."
    )

    data_args.add_argument(
        "--is-balance-subset",
        default=False,
        action="store_true",
        help="Whether to subset the data in a balanced rather than stratified way. "
        "Only works if final subset size is smaller than under sampled balanced data and not multi label. "
        "Only when train_size != -1",
    )
    data_args.add_argument(
        "--balance-data-mode",
        default=None,
        choices=[None, "undersample", "oversample"],
        help="How and whether to balance the training dataset. "
        "Applied after potential subsetting).",
    )
    data_args.add_argument(
        "--train-sizes",
        default=[-1],
        nargs="+",
        type=float,
        help="Sizes of the training set to evaluate for. Percentage if <= 1 else size. "
        "-1 means all. Will run evaluation for each.",
    )

    train_args.add_argument(
        "--is-sklearn",
        default=False,
        action="store_true",
        help="Whether to use sklearn instead of pytorch logistic regression.",
    )
    train_args.add_argument(
        "--is-no-progress-bar",
        default=False,
        action="store_true",
        help="Whether to disable progressbar.",
    )
    train_args.add_argument(
        "--is-balance-loss",
        default=False,
        action="store_true",
        help="Whether to use per class / label balanced loss during training of the probe."
        "If so you will also balance the validation set for hyperparam tuning.",
    )
    train_args.add_argument(
        "--is-validation",
        default=False,
        action="store_true",
        help="Whether to use a validation set => hyperparameter tuning. If yes you will use:"
        "valid data if available split else 10% of the original training data.",
    )
    # NB : using a validation set is not realistic when train_size small (because validation would be larger than train)
    train_args.add_argument(
        "--n-hyper-param",
        type=int,
        default=10,
        help="Number of parameters to sample when performing validation.",
    )

    torch_args.add_argument(
        "--n-epochs",
        default=[100],
        nargs="+",
        type=int,
        help="Number of total epochs to run. There should be one value per training size.",
    )
    torch_args.add_argument(
        "--is-batchnorm",
        default=False,
        action="store_true",
        help="optionally add a batchnorm layer before the linear classifier if not tuning over.",
    )
    torch_args.add_argument(
        "--is-mlp",
        default=False,
        action="store_true",
        help="use MLP probe instead of linear.",
    )
    torch_args.add_argument(
        "--is-mlpS",
        default=False,
        action="store_true",
        help="use MLP probe instead of linear.",
    )
    torch_args.add_argument(
        "--lr", default=0.3, type=float, help="learning rate for the model if not tuning over." 
                                              "This is lr for batch_size 256"
    )
    torch_args.add_argument("--batch-size", default=256, type=int, help="batch size if not tuning over.")
    torch_args.add_argument(
        "--weight-decay", default=1e-6, type=float, help="weight decay if not tuning over."
    )
    torch_args.add_argument(
        "--n-gpus", default=1, type=int, help="Number of gpus to use"
    )
    torch_args.add_argument(
        "--torch-validate-param",
        default=["lr", "weight_decay"],
        nargs="+",
        choices=["lr", "batch_size", "weight_decay", "is_batchnorm"],
        help="Parameters to validate over if using validation set.",
    )
    torch_args.add_argument(
        "--no-wandb",
        default=False,
        action="store_true",
        help="Whether not to use weights and biases.",
    )
    torch_args.add_argument(
        "--is-monitor-test",
        default=False,
        action="store_true",
        help="Whether to monitor test performance.",
    )
    torch_args.add_argument(
        "--is-lars",
        default=False,
        action="store_true",
        help="Whether to use the LARS optimizer, which can be helpful in large batch settings.",
    )
    torch_args.add_argument(
        "--is-adamw",
        default=False,
        action="store_true",
        help="Whether to use the AdamW optimizer.",
    )

    sk_args.add_argument(
        "--C",
        default=1.0,
        type=float,
        help="regularization (smaller is more) if not tuning over.",
    )
    sk_args.add_argument(
        "--is-svm",
        default=False,
        action="store_true",
        help="Whether to use linear SVM instead of logistic regression.",
    )
    sk_args.add_argument(
        "--no-scale-features",
        default=False,
        action="store_true",
        help="Whether not to min max scale the features before classifier. Note that still linear.",
    )
    sk_args.add_argument(
        "--sk-validate-param",
        default=["C"],
        nargs="+",
        choices=["C", "penalty"],
        help="Parameters to validate over if using validation set.",
    )

    cfg = parser.parse_args()

    assert len(cfg.train_sizes) == len(cfg.n_epochs)

    # setting desired type
    cfg.feature_path = Path(cfg.feature_path)
    cfg.out_path = Path(cfg.out_path)

    # adding values to fill
    cfg.__dict__["curr_train_size"] = None
    cfg.__dict__["curr_n_epoch"] = None
    cfg.__dict__["type_of_target"] = None

    # double negatives -> pos
    cfg.__dict__["is_scale_features"] = not cfg.no_scale_features


    logging.info(f"Configs: {cfg}")

    main(cfg)