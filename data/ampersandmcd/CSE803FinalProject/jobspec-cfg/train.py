import os
from argparse import ArgumentParser
from models import SRCNN, VDSR, SRResNet
from data import ERA5DataModule
import pytorch_lightning as pl
import wandb

from utils import ImageVisCallback

wandb.init(project='cv-proj', entity="cv803f21-superres")


def main(args):
    # configure data module
    e = ERA5DataModule(args={
        "pool_size": args.pool_size,
        "batch_size": args.batch_size,
        "patch_size": args.patch_size
    })
    train_dl, val_dl = e.train_dataloader(), e.val_dataloader()

    # input channels controls which channels we use as predictors
    # output channels controls which channels we use as targets, i.e., loss signal
    # channel 0 corresponds to t2m and channel 1 corresponds to tp
    # e.g., input_channels=[0, 1], output_channels=[1] predicts tp @ HR using t2m AND tp @ LR
    # e.g., input_channels=[1],    output_channels=[1] predicts tp @ HR using ONLY tp @ LR
    # ...etc.
    args.model = args.model if hasattr(args, "model") else "SRCNN"
    if args.model.lower() == "vdsr":
        print("Constructing VDSR")
        model = VDSR(input_channels=[0, 1], output_channels=[0, 1], lr=args.lr, decayRate=args.decay_Rate)
    elif args.model.lower() == "srresnet":
        print("Constructing SRResNet")
        model = SRResNet(input_channels=[0, 1], output_channels=[0, 1], lr=args.lr)
    elif args.model.lower() == "srcnn":
        print("Constructing SRCNN")
        model = SRCNN(input_channels=[0, 1], output_channels=[0, 1], lr=args.lr)
    else:
        raise ValueError("Invalid model architecture.")

    # Wandb logging
    wandb_logger = pl.loggers.WandbLogger(project='cv-proj')
    wandb_logger.watch(model, log_freq=500)

    trainer: pl.Trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = wandb_logger
    trainer.callbacks.append(ImageVisCallback(val_dl))

    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--model', default="SRCNN", type=str, help="Model to train")
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size to train with")
    parser.add_argument('--pool_size', default=4, type=int, help="Super-resolution factor")
    parser.add_argument('--patch_size', default=64, type=int, help="Image patch size to super-resolve")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--decay_Rate", default=1, typep=float, help="Exponential decay rate")
    args = parser.parse_args()

    main(args)
