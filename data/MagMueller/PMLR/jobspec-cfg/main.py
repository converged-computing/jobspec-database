
# %% Imports

from ast import arg
from pytorch_lightning.loggers import WandbLogger

import time

from model import deep_GNN
import torch
from torch import device, nn
import torch
import os
from utils.dataset import H5GraphDataset, H5ImageDataset

import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from model import LitModel
import numpy as np
from torch.utils.data import ConcatDataset
from pytorch_lightning.callbacks import ModelCheckpoint

from coRNN import coRNN, coRNN2
from deep_GNN import deep_GNN
import hydra
from omegaconf import DictConfig, OmegaConf
import uuid


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=5,
        mode='min'
    )
    run_name = ""
    if cfg.eval:
        run_name = "eval_model"
        if cfg.stupid:
            run_name = "eval_stupid"
    else:
        unique_id = uuid.uuid4()
        run_name = cfg.model.name + "_hid:" + str(cfg.model.n_hid) + "_epoch:" + str(cfg.epochs) + "_" + str(unique_id)

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb_logger = WandbLogger(project='PMLR', log_model="all", name=run_name, config=config)
    # wandb_logger.experiment.config.update(config)

    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=cfg.epochs,
        strategy=cfg.env.strategy,
        num_nodes=cfg.env.nodes,
        devices=cfg.env.devices,
        precision=32,
        # limit_train_batches=12,
        # limit_val_batches=12,
        accumulate_grad_batches=4,
        profiler="simple",
        callbacks=[checkpoint_callback],
        limit_val_batches=200,
        limit_test_batches=40,
    )
    print(f"Number of devices: {trainer.num_devices}")

    # for normalizing

    # m = torch.as_tensor((time_means - means)/stds)[:, 0:cfg.height].unsqueeze(0)
    # std = torch.tensor(stds).unsqueeze(0)

    # wandb_logger.watch(model, log='all', log_freq=100)

    if cfg.model.name == "coRNN":
        model = coRNN(**cfg.model)
        datasets = {
            "train": ConcatDataset([H5ImageDataset(os.path.join(cfg.env.train_folder, f"{year}.h5"), cfg=cfg) for year in cfg.env.years]),
            "val": H5ImageDataset(cfg.env.val_file, cfg=cfg)
        }
    elif cfg.model.name == "coRNN2":
        model = coRNN2(**cfg.model)
        datasets = {
            "train": ConcatDataset([H5ImageDataset(os.path.join(cfg.env.train_folder, f"{year}.h5"), cfg=cfg) for year in cfg.env.years]),
            "val": H5ImageDataset(cfg.env.val_file, cfg=cfg)
        }
    elif cfg.model.name == "deep_coRNN":
        model = deep_GNN(**cfg.model)
        datasets = {
            "train": ConcatDataset([H5GraphDataset(os.path.join(cfg.env.train_folder, f"{year}.h5"), cfg=cfg) for year in cfg.env.years]),
            "val": H5GraphDataset(cfg.env.val_file, cfg=cfg)
        }
    else:
        raise ValueError("Model name not recognized")

    if cfg.eval:
        # use script download_artifact.py to download the model
        path = "artifacts/model-4yobs5ix:v9" + "/model.ckpt"

        model = LitModel.load_from_checkpoint(checkpoint_path=path, datasets=datasets,  model=model, config=cfg)

        print(f"autoregressive step is {model.count_autoreg_steps}")
        print(f"Model evaluation")
        trainer.test(model)
        return
    else:
        model = LitModel(datasets=datasets,  model=model, cfg=cfg)

        wandb_logger.watch(model, log='all', log_freq=100)
        trainer.fit(model)


# %%
if __name__ == "__main__":
    main()
