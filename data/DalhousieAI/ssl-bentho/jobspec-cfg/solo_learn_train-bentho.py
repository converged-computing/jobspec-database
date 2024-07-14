#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from solo.data.pretrain_dataloader import (
    build_transform_pipeline,
    prepare_n_crop_transform,
)
from solo.methods import (
    BYOL,
    DINO,
    MAE,
    BarlowTwins,
    MoCoV2Plus,
    MoCoV3,
    SimCLR,
    SimSiam,
)

import benthic_data_classes.datasets
from utils.benthicnet.io import read_csv

METHODS = {
    "bt": BarlowTwins,
    "dino": DINO,
    "simclr": SimCLR,
    "mocov2+": MoCoV2Plus,
    "mocov3": MoCoV3,
    "mae": MAE,
    "simsiam": SimSiam,
    "byol": BYOL,
}


def get_df(in_path):
    df = read_csv(
        fname=in_path, expect_datetime=False, index_col=None, low_memory=False
    )
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Parameters for SSL benthic habitat project"
    )

    # Required parameters
    parser.add_argument(
        "--ssl_cfg", type=str, required=True, help="set cfg file for SSL"
    )
    parser.add_argument("--nodes", type=int, required=True, help="number of nodes")
    parser.add_argument(
        "--gpus", type=int, required=True, help="number of gpus per node"
    )
    parser.add_argument("--method", type=str, required=True, help="select SSL method")
    # Other parameters
    parser.add_argument(
        "--aug_stack_cfg",
        type=str,
        default="simclr_aug_stack.cfg",
        help="set cfg file for augmentations",
    )
    parser.add_argument(
        "--csv_file_path",
        type=str,
        default="./data_csv/benthicnet_unlabelled_nn.csv",
        help="set path to csv file",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--name",
        type=str,
        default="self-supervised_learning",
        help="set name for the run",
    )

    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    ssl_csv_path = args.csv_file_path
    ssl_csv = get_df(ssl_csv_path)

    # common parameters for all methods
    # some parameters for extra functionally are missing, but don't mind this for now.

    ssl_cfg_name = args.ssl_cfg

    with open("./ssl_cfgs/" + ssl_cfg_name, encoding="utf-8") as f:
        ssl_cfg = f.read()

    kwargs = json.loads(ssl_cfg)
    cfg = OmegaConf.create(kwargs)

    model = METHODS[args.method](cfg)

    if args.method == "dino":
        with open("./ssl_cfgs/aug_stacks/dino_first_global.cfg", encoding="utf-8") as f:
            dino_first_global_cfg = f.read()

        with open(
            "./ssl_cfgs/aug_stacks/dino_second_global.cfg", encoding="utf-8"
        ) as f:
            dino_second_global_cfg = f.read()

        with open("./ssl_cfgs/aug_stacks/dino_local.cfg", encoding="utf-8") as f:
            dino_local_cfg = f.read()

        dino_first_global_kwargs = json.loads(dino_first_global_cfg)
        dino_second_global_kwargs = json.loads(dino_second_global_cfg)
        dino_local_kwargs = json.loads(dino_local_cfg)

        dino_first_global_cfg = OmegaConf.create(dino_first_global_kwargs)
        dino_second_global_cfg = OmegaConf.create(dino_second_global_kwargs)
        dino_local_cfg = OmegaConf.create(dino_local_kwargs)

        dino_first_global_transform = build_transform_pipeline(
            "custom", dino_first_global_cfg
        )
        dino_second_global_transform = build_transform_pipeline(
            "custom", dino_second_global_cfg
        )
        dino_local_transform = build_transform_pipeline("custom", dino_local_cfg)

        dino_first_global_transform = prepare_n_crop_transform(
            [dino_first_global_transform],
            num_crops_per_aug=[int(kwargs["data"]["num_large_crops"] / 2)],
        )
        dino_second_global_transform = prepare_n_crop_transform(
            [dino_second_global_transform],
            num_crops_per_aug=[int(kwargs["data"]["num_large_crops"] / 2)],
        )
        dino_local_transform = prepare_n_crop_transform(
            [dino_local_transform],
            num_crops_per_aug=[kwargs["data"]["num_small_crops"]],
        )

        train_dataset = benthic_data_classes.datasets.BenthicNetDatasetSSL(
            ssl_csv,
            [
                dino_first_global_transform,
                dino_second_global_transform,
                dino_local_transform,
            ],
        )

    else:
        # we first prepare our single transformation pipeline

        aug_stack_name = args.aug_stack_cfg

        with open("./ssl_cfgs/aug_stacks/" + aug_stack_name, encoding="utf-8") as f:
            aug_stack_cfg = f.read()

        transform_kwargs = json.loads(aug_stack_cfg)
        transform_cfg = OmegaConf.create(transform_kwargs)

        transform = build_transform_pipeline("custom", transform_cfg)

        # then, we wrap the pipepline using this utility function
        # to make it produce an arbitrary number of crops
        transform = prepare_n_crop_transform(
            [transform], num_crops_per_aug=[kwargs["data"]["num_large_crops"]]
        )

        train_dataset = benthic_data_classes.datasets.BenthicNetDatasetSSL(
            ssl_csv, transform
        )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=kwargs["optimizer"]["batch_size"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=kwargs["num_workers"],
    )

    run_name = args.name

    # Set up callbacks
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    directory_path = os.path.join("checkpoints", timestamp)

    csv_logger = pl_loggers.CSVLogger(
        "logs", name=run_name + "_logs", version=timestamp
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=directory_path,
        filename=args.name + "_{epoch:03d}",
        save_top_k=1,
        mode="min",
        every_n_epochs=cfg.max_epochs,
        save_weights_only=True,
    )

    # automatically log our learning rate
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # checkpointer can automatically log your parameters,
    # but we need to wrap it on a Namespace object

    callbacks = [checkpoint_callback, lr_monitor]

    # Adapt for pytorch lightning 2.0+

    trainer = Trainer(
        logger=csv_logger,
        callbacks=callbacks,
        strategy="auto",
        accelerator="auto",
        log_every_n_steps=200,
        num_nodes=args.nodes,
        devices=args.gpus,
        max_epochs=cfg.max_epochs,
    )

    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
