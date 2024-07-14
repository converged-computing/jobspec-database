import argparse
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional.classification import binary_accuracy, binary_average_precision
from torchvision.transforms.functional import hflip

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from src.data_loader import get_dataloaders
from src.nn.model_collection import MODEL_DICT


class Classifier(pl.LightningModule):
    def __init__(self, model: str, freeze_backbone: bool, optimizer: str, learning_rate: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.classifier = MODEL_DICT[model](freeze_backbone=freeze_backbone)
        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        dire, label = batch
        if np.random.rand() < 0.5:  # 50% chance for horizontal flip
            dire = hflip(dire)
        logit = self.classifier(dire)
        loss = self.loss(logit, label)
        acc = binary_accuracy(logit.argmax(axis=1), label)
        ap = binary_average_precision(logit[:, 1], label)
        metrics = {"train_loss": loss, "train_acc": acc, "train_ap": ap}
        self.log_dict(metrics)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        dire, label = batch
        logit = self.classifier(dire)
        loss = self.loss(logit, label)
        acc = binary_accuracy(logit.argmax(axis=1), label)
        ap = binary_average_precision(logit[:, 1], label)
        metrics = {"val_loss": loss, "val_acc": acc, "val_ap": ap}
        self.log_dict(metrics)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        dire, label = batch
        logit = self.classifier(dire)
        loss = self.loss(logit, label)
        acc = binary_accuracy(logit.argmax(axis=1), label)
        ap = binary_average_precision(logit[:, 1], label)
        metrics = {"test_loss": loss, "test_acc": acc, "test_ap": ap}
        self.log_dict(metrics)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = Adam if self.hparams.optimizer == "Adam" else SGD
        optimizer = optimizer(self.classifier.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=3)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_acc"}


def main(args: argparse.Namespace) -> None:
    torch.set_float32_matmul_precision("medium")
    seed_everything(33914, workers=True)

    # Setup Weights & Biases
    wandb_logger = WandbLogger(name=args.name, project="Training", entity="latent-dire", config=vars(args))

    # Load the data
    train_loader, val_loader, test_loader = get_dataloaders(
        root=args.data_dir,
        model=args.model,
        type=args.data_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Setup callbacks
    early_stop = EarlyStopping(monitor="val_acc", mode="max", min_delta=0.0, patience=5, verbose=True)
    checkpoint = ModelCheckpoint(save_top_k=2, monitor="val_acc", mode="max", dirpath=f"models/{args.name}")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    clf = Classifier(args.model, args.freeze_backbone, args.optimizer, args.learning_rate)
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",  # use all available GPUs
        min_epochs=1,
        max_epochs=args.max_epochs,
        callbacks=[early_stop, checkpoint, lr_monitor],
        val_check_interval=25,
        # deterministic=True,  # slower, but reproducable: https://lightning.ai/docs/pytorch/stable/common/trainer.html#reproducibility
        precision="16-mixed",
        logger=wandb_logger,
        log_every_n_steps=5,
        fast_dev_run=args.dev_run,
    )
    trainer.fit(clf, train_loader, val_loader)
    trainer.test(clf, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dev_run", action="store_true", help="Whether to run a test run.")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        required="-d" not in sys.argv and "--dev-run" not in sys.argv,
        help="A descriptive name for the run, for wandb and checkpoint directory.",
    )
    parser.add_argument("--model", type=str, required=True, choices=["resnet50_pixel", "resnet50_latent"])
    parser.add_argument("--data_type", type=str, required=True, choices=["images", "latent"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--freeze_backbone", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"], help="Optimizer to use")
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for the data loader.")
    # parser.add_argument("--latent", type=bool, default=False, help="Whether to use Latent DIRE")
    # parser.add_argument("--use_early_stopping", type=int, default=1, help="Whether to use early stopping.")
    args = parser.parse_args()

    # Check arguments
    assert args.model in ["resnet50_latent", "resnet50_pixel", "mlp", "cnn"]
    assert args.batch_size > 0
    assert args.max_epochs > 0
    # if args.latent:
    #    assert args.model in ["mlp", "cnn", "resnet50_latent"]
    # else:
    #    assert args.model == "resnet50_pixel"

    main(args)
