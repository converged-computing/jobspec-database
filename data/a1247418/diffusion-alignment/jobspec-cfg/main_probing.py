# Adapted from: https://github.com/LukasMut/human_alignment/blob/main/main_probing.py

import argparse
import os
import pickle
import time
from typing import Any, Callable, Dict, Iterator, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA

import utils
from helpers import load_features

Array = np.ndarray
Tensor = torch.Tensor
FrozenDict = Any


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data_root", type=str, help="path/to/things")
    aa("--dataset", type=str, help="Which dataset to use", default="things")
    aa("--model", type=str, nargs="+", help="Which model to use. For multiple models, features will be stacked.")
    aa(
        "--module",
        type=str,
        default="penultimate",
        nargs="+",
        help="DNN module for which to learn a linear transform. For multiple modules, features will be stacked.",
    )
    aa(
        "--source",
        type=str,
        default="torchvision",
        choices=[
            "custom",
            "torchvision",
            "diffusion",
        ],
    )
    aa(
        "--n_objects",
        type=int,
        help="Number of object categories in the data",
        default=1854,
    )
    aa(
        "--n_folds",
        type=int,
        default=3,
        choices=[2, 3, 4, 5],
        help="Number of folds in k-fold cross-validation.",
    )
    aa("--optim", type=str, default="Adam", choices=["Adam", "AdamW", "SGD"])
    aa("--learning_rate", type=float, default=1e-4)
    aa(
        "--lmbda",
        type=float,
        default=1e-3,
        help="Relative contribution of the regularization term",
        choices=[10.0, 1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    )
    aa(
        "--batch_size",
        type=int,
        default=256,
        help="Use power of 2 for running optimization on GPU",
        choices=[64, 128, 256, 512, 1024],
    )
    aa(
        "--epochs",
        type=int,
        help="Maximum number of epochs to perform finetuning",
        default=100,
    )
    aa(
        "--burnin",
        type=int,
        help="Minimum number of epochs to perform finetuning",
        default=3,
    )
    aa(
        "--patience",
        type=int,
        help="number of checks with no improvement after which training will be stopped",
        default=5,
    )
    aa("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    aa(
        "--num_processes",
        type=int,
        default=4,
        help="Number of devices to use for performing distributed training on CPU",
    )
    aa(
        "--use_bias",
        action="store_true",
        help="whether or not to use a bias for the naive transform",
    )
    aa("--probing_root", type=str, help="path/to/probing")
    aa("--log_dir", type=str, help="directory to checkpoint transformations")
    aa("--rnd_seed", type=int, default=42, help="random seed for reproducibility")
    aa(
        "--overwrite",
        action="store_true",
        help="whether or not to overwrite existing results",
    )
    aa(
        "--cc0",
        action="store_true",
        help="whether or not to use CC0 files",
    )
    aa("--pca", type=int, help="dimensionality of the PCA", default=None)
    aa("--subfolder", type=str, help="subfolder for features", default="things_data")
    args = parser.parse_args()
    return args


def create_optimization_config(args) -> Tuple[FrozenDict, FrozenDict]:
    """Create frozen config dict for optimization hyperparameters."""
    optim_cfg = dict()
    optim_cfg["optim"] = args.optim
    optim_cfg["lr"] = args.learning_rate
    optim_cfg["lmbda"] = args.lmbda
    optim_cfg["n_folds"] = args.n_folds
    optim_cfg["batch_size"] = args.batch_size
    optim_cfg["max_epochs"] = args.epochs
    optim_cfg["min_epochs"] = args.burnin
    optim_cfg["patience"] = args.patience
    optim_cfg["use_bias"] = args.use_bias
    optim_cfg["ckptdir"] = os.path.join(args.log_dir, "+".join(args.model), "+".join(args.module))
    return optim_cfg


def get_batches(triplets: Tensor, batch_size: int, train: bool) -> Iterator:
    batches = DataLoader(
        dataset=triplets,
        batch_size=batch_size,
        shuffle=True if train else False,
        num_workers=8,
        drop_last=False,
        pin_memory=True if train else False,
    )
    return batches


def get_callbacks(optim_cfg: FrozenDict, steps: int = 20) -> List[Callable]:
    if not os.path.exists(optim_cfg["ckptdir"]):
        os.makedirs(optim_cfg["ckptdir"])
        print("\nCreating directory for checkpointing...\n")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=optim_cfg["ckptdir"],
        filename="ooo-finetuning-epoch{epoch:02d}-val_loss{val/loss:.2f}",
        auto_insert_metric_name=False,
        every_n_epochs=steps,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        mode="min",
        patience=optim_cfg["patience"],
        verbose=True,
        check_finite=True,
    )
    callbacks = [checkpoint_callback, early_stopping]
    return callbacks


def get_mean_cv_acc(
    cv_results: Dict[str, List[float]], metric: str = "test_acc"
) -> float:
    avg_val_acc = np.mean([vals[0][metric] for vals in cv_results.values()])
    return avg_val_acc


def get_mean_cv_loss(
    cv_results: Dict[str, List[float]], metric: str = "test_loss"
) -> float:
    avg_val_loss = np.mean([vals[0][metric] for vals in cv_results.values()])
    return avg_val_loss


def make_results_df(
    columns: List[str],
    probing_acc: float,
    probing_loss: float,
    ooo_choices: Array,
    model_name: str,
    module_name: str,
    source: str,
    lmbda: float,
    optim: str,
    lr: float,
    n_folds: int,
    bias: bool,
) -> pd.DataFrame:
    probing_results_current_run = pd.DataFrame(index=range(1), columns=columns)
    probing_results_current_run["model"] = model_name
    probing_results_current_run["probing"] = probing_acc
    probing_results_current_run["cross-entropy"] = probing_loss
    # probing_results_current_run["choices"] = [ooo_choices]
    probing_results_current_run["module"] = module_name
    probing_results_current_run["source"] = source
    probing_results_current_run["l2_reg"] = lmbda
    probing_results_current_run["optim"] = optim.lower()
    probing_results_current_run["lr"] = lr
    probing_results_current_run["n_folds"] = n_folds
    probing_results_current_run["bias"] = bias
    return probing_results_current_run


def save_results(
    args, probing_acc: float, probing_loss: float, ooo_choices: Array
) -> None:
    out_path = os.path.join(args.probing_root, "results")
    results_file = os.path.join(out_path, "probing_results.pkl")
    if args.pca:
        results_file = results_file.replace(".pkl", f"_pca{args.pca}.pkl")

    if not os.path.exists(out_path):
        print("\nCreating results directory...\n")
        os.makedirs(out_path)

    if os.path.isfile(results_file):
        print(
            "\nFile for probing results exists.\nConcatenating current results with existing results file...\n"
        )


        try:
            probing_results_overall = pd.read_pickle(results_file)
        except EOFError as e:
            print(repr(e), "for loading path:", results_file)
            print("Trying again after random delay 1/3..")
            time.sleep(np.random.randint(5, 60))
            try:
                probing_results_overall = pd.read_pickle(results_file)
            except EOFError as e:
                print(repr(e), "for loading path:", results_file)
                print("Trying again after random delay 2/3..")
                time.sleep(np.random.randint(5, 60))
                try:
                    probing_results_overall = pd.read_pickle(results_file)
                except EOFError as e:
                    print(repr(e), "for loading path:", results_file)
                    print("Trying again after random delay 3/3..")
                    time.sleep(np.random.randint(5, 60))
                    probing_results_overall = pd.read_pickle(results_file)

        probing_results_current_run = make_results_df(
            columns=probing_results_overall.columns.values,
            probing_acc=probing_acc,
            probing_loss=probing_loss,
            ooo_choices=ooo_choices,
            model_name="+".join(args.model),
            module_name="+".join(args.module),
            source=args.source,
            lmbda=args.lmbda,
            optim=args.optim,
            lr=args.learning_rate,
            n_folds=args.n_folds,
            bias=args.use_bias,
        )
        probing_results = pd.concat(
            [probing_results_overall, probing_results_current_run],
            axis=0,
            ignore_index=True,
        )
        probing_results.to_pickle(results_file)
    else:
        print("\nCreating file for probing results...\n")
        columns = [
            "model",
            "probing",
            "cross-entropy",
            # "choices",
            "module",
            "source",
            "l2_reg",
            "optim",
            "lr",
            "n_folds",
            "bias",
        ]
        probing_results = make_results_df(
            columns=columns,
            probing_acc=probing_acc,
            probing_loss=probing_loss,
            ooo_choices=ooo_choices,
            model_name="+".join(args.model),
            module_name="+".join(args.module),
            source=args.source,
            lmbda=args.lmbda,
            optim=args.optim,
            lr=args.learning_rate,
            n_folds=args.n_folds,
            bias=args.use_bias,
        )
        probing_results.to_pickle(results_file)


def run(
    features: Array,
    data_root: str,
    n_objects: int,
    device: str,
    optim_cfg: FrozenDict,
    rnd_seed: int,
    num_processes: int,
    pca: Optional[int] = None
) -> Tuple[Dict[str, List[float]], Array]:
    """Run optimization process."""
    callbacks = get_callbacks(optim_cfg)
    triplets = utils.probing.load_triplets(data_root)

    features = (
        features - features.mean()
    ) / features.std()  # subtract mean and normalize by standard deviation
    optim_cfg["sigma"] = 1e-3
    objects = np.arange(n_objects)
    # Perform k-fold cross-validation with k = 3 or k = 4
    kf = KFold(n_splits=optim_cfg["n_folds"], random_state=rnd_seed, shuffle=True)
    cv_results = {}
    ooo_choices = []
    for k, (train_idx, _) in tqdm(enumerate(kf.split(objects), start=1), desc="Fold"):
        train_objects = objects[train_idx]
        # partition triplets into disjoint object sets
        triplet_partitioning = utils.probing.partition_triplets(
            triplets=triplets,
            train_objects=train_objects,
        )
        train_triplets = utils.probing.TripletData(
            triplets=triplet_partitioning["train"],
            n_objects=n_objects,
        )
        val_triplets = utils.probing.TripletData(
            triplets=triplet_partitioning["val"],
            n_objects=n_objects,
        )
        train_batches = get_batches(
            triplets=train_triplets,
            batch_size=optim_cfg["batch_size"],
            train=True,
        )
        val_batches = get_batches(
            triplets=val_triplets,
            batch_size=optim_cfg["batch_size"],
            train=False,
        )

        if pca is not None:
            print(f"Performing PCA ({pca})")
            pca_algo = PCA(n_components=min(pca, min(features.shape[0], features.shape[1])))
            pca_algo.fit(features[objects])
            features_transformed = pca_algo.transform(features)
            print("  New dimensionality:", features_transformed.shape)
        else:
            features_transformed = features

        linear_probe = utils.probing.Linear(
            features=features_transformed,
            optim_cfg=optim_cfg,
        )
        trainer = Trainer(
            accelerator=device,
            callbacks=callbacks,
            strategy="ddp",
            max_epochs=optim_cfg["max_epochs"],
            min_epochs=optim_cfg["min_epochs"],
            devices=num_processes if device == "cpu" else "auto",
            enable_progress_bar=True,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
        )
        trainer.fit(linear_probe, train_batches, val_batches)
        val_performance = trainer.test(
            linear_probe,
            dataloaders=val_batches,
        )
        predictions = trainer.predict(linear_probe, dataloaders=val_batches)
        predictions = torch.cat(predictions, dim=0).tolist()
        ooo_choices.append(predictions)
        cv_results[f"fold_{k:02d}"] = val_performance
    transformation = linear_probe.transform_w.data.detach().cpu().numpy()
    if optim_cfg["use_bias"]:
        bias = linear_probe.transform_b.data.detach().cpu().numpy()
        transformation = np.concatenate((transformation, bias[:, None]), axis=1)
    ooo_choices = np.concatenate(ooo_choices)
    return ooo_choices, cv_results, transformation


if __name__ == "__main__":
    # parse arguments
    args = parseargs()

    out_path = os.path.join(
        args.probing_root,
        "results",
        args.source,
        "+".join(args.model),
        "+".join(args.module),
        str(args.n_folds),
        str(args.lmbda),
        args.optim.lower(),
        str(args.learning_rate),
    )
    out_file_path = os.path.join(out_path, "transform.npy")

    if os.path.exists(out_file_path) and not args.overwrite:
        print(
            "\nFile for linear transform exists. Skipping."
        )
    else:
        # seed everything for reproducibility of results
        seed_everything(args.rnd_seed, workers=True)

        # TODO remove
        #features = load_features(args.probing_root, subfolder=args.subfolder, cc0=args.cc0)#, pca=args.pca)

        # Stack multiple representations, if given
        model_features = []
        for model in args.model:
            for module in args.module:
                # TODO remove
                #model_features.append(features[args.source][model][module])
                model_features.append(load_features(
                    path=args.probing_root,
                    subfolder=args.subfolder,
                    model_name=model,
                    module_type=module,
                    source=args.source,
                    cc0=args.cc0,
                ))
        model_features = np.concatenate(model_features, axis=1)

        if model_features.dtype == np.float16:
            print("   Converting to float32.")
            model_features = np.float32(model_features)

        optim_cfg = create_optimization_config(args)
        ooo_choices, cv_results, transform = run(
            features=model_features,
            data_root=args.data_root,
            n_objects=args.n_objects,
            device=args.device,
            optim_cfg=optim_cfg,
            rnd_seed=args.rnd_seed,
            num_processes=args.num_processes,
            pca=args.pca
        )
        avg_cv_acc = get_mean_cv_acc(cv_results)
        avg_cv_loss = get_mean_cv_loss(cv_results)

        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)

        try:
            with open(os.path.join(out_path, "result.npy"), "wb") as f:
                pickle.dump(cv_results, f)
        except Exception as e:
            print("Could not save individual result.", repr(e))

        print("Odd-one-out accuracy:", avg_cv_acc)
        save_results(
            args, probing_acc=avg_cv_acc, probing_loss=avg_cv_loss, ooo_choices=ooo_choices
        )

        with open(out_file_path, "wb") as f:
            np.save(file=f, arr=transform)


