import importlib
import data_utils
importlib.reload(data_utils)
from data_utils import PhonemeDataset
from mlp_mixer import MLPMixer
from functools import partial

import optuna
import numpy as np
import torch 
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import pytorch_lightning as pl
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

torch.autograd.set_detect_anomaly(True)

def get_data(batch_size=32, transform=None):
    train_loader = DataLoader(
        PhonemeDataset(
            data_filename='../Data/Phoneme/train_X.npy',
            label_filename='../Data/Phoneme/train_y.npy',
            transform=transform
        ), 
        batch_size=batch_size, 
        shuffle=True
    )

    val_loader = DataLoader(
        PhonemeDataset(
            data_filename='../Data/Phoneme/valid_X.npy',
            label_filename='../Data/Phoneme/valid_y.npy',
            transform=None
        ), 
        batch_size=batch_size, 
        shuffle=False
    )

    return train_loader, val_loader

def objective(trial: optuna.trial.Trial, patch_class) -> float:
    num_classes = 39
    padded_length = 220
    batch_size = 128
    transform = None
    epochs = 1000
    p_dropout = 0.5

    # Generate the model
    model = MLPMixer(
        num_classes=num_classes,
        patch_class=patch_class,
        padded_length=padded_length,
        p_dropout=p_dropout,
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        num_blocks=trial.suggest_int("num_blocks", 1, 6),
        patch_size=trial.suggest_categorical("patch_size", [2, 4, 5, 10, 11, 20, 22, 44, 55, 110]),
        hidden_dim=trial.suggest_int("hidden_dim", 16, 1024),
        tokens_mlp_dim=trial.suggest_int("tokens_mlp_dim", 16, 2048),
        channels_mlp_dim=trial.suggest_int("channels_mlp_dim", 16, 2048)
    )

    # Generate the dataloaders
    train_loader, valid_loader = get_data(batch_size=batch_size, transform=transform)

    # Train
    callbacks = [EarlyStopping(monitor="val_loss", patience=25, mode="min")]

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        callbacks=callbacks
    )
    
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader
    )

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
    patch_classes = ["cyclical1d"] #["sequential1d", "random1d", "cyclical1d"]

    for patch_class in patch_classes:
        print(f"\n \n \n Optimizing for patch class: {patch_class}")

        # Create optuna study
        objective = partial(objective, patch_class=patch_class)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=200, timeout=(16*60*60))

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("Value: {}".format(trial.value))

        print("Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))