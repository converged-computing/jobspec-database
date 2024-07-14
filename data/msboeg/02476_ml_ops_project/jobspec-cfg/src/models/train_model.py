import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime

import gcsfs
import matplotlib.pyplot as plt
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from torch import nn, optim
from torchvision import datasets

from src.data.FlowerDataset import FlowerDataset
from src.models.task import fit_detector, get_args
from src.models.ViT import ViT


def main():
    # Training settings
    args = get_args()

    current_time = datetime.now().strftime("%Y%m%d%H%M%S")

    logger = None
    if args.wandb_api_key:
        logger = pl_loggers.WandbLogger(
            name="trained_models",
            version=current_time,
            project="ml_ops_project",
            entity="ml_ops_team10",
            config=args,
        )
        wandb.login(key=args.wandb_api_key)
        print("Using wandb for logging.")
    else:
        logger = pl_loggers.TensorBoardLogger(
            args.model_dir if args.model_dir else "tb_logs",
            name="trained_models",
            version=current_time,
        )
        print("No wandb API key provided. Using local TensorBoard.")

    if args.data_path.startswith("gs://"):
        print("Downloading data from Google Cloud Storage")
        gcsfs.GCSFileSystem().get(args.data_path, "tmp", recursive=True)
        args.data_path = "tmp"
        print("Data downloaded to ", args.data_path)

    # Load the training data
    train_set = FlowerDataset(args.data_path, "224x224", "train")
    val_set = FlowerDataset(args.data_path, "224x224", "val")
    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = ViT(**vars(args))
    trainer = Trainer(
        logger=logger,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        enable_checkpointing=True if args.model_dir else False,
        default_root_dir=args.model_dir,
        auto_select_gpus=args.auto_select_gpus,
        log_every_n_steps=2,
        gpus=args.gpus,
        strategy="ddp" if args.gpus > 1 else None,
    )

    trainer.fit(model, trainloader, valloader)

    model_copy = deepcopy(model)
    model_copy.ViT[1] = nn.Identity()

    feature_extractor = nn.Sequential(model_copy, nn.Flatten())

    drift_detector = fit_detector(trainloader, feature_extractor)

    if args.model_dir:
        foldername = os.path.join(args.model_dir, "trained_models", current_time)

        if not os.path.exists(foldername):
            os.makedirs(foldername)

        torch.save(drift_detector, os.path.join(foldername, "drift_detector.pt"))


if __name__ == "__main__":
    main()
