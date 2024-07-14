import hydra
from pytorch_lightning import Trainer, callbacks, loggers, seed_everything
from scaling_model.models.utils import PredictionWriter
from scaling_model.models.painn_lightning import PaiNNforQM9
from scaling_model.data.data_module import (
    QM9DataModule,
    BaselineDataModule,
)

from lightning.pytorch.profilers import PyTorchProfiler
from torch.profiler import ProfilerActivity
import pytorch_lightning as pl
import torch
from lightning.pytorch.profilers import AdvancedProfiler
import os

torch.set_float32_matmul_precision("medium")


@hydra.main(
    config_path="configs",
    config_name="lightning_config",
    version_base=None,
)
def main(cfg):
    seed_everything(cfg.seed)
    if not os.path.exists(cfg.logger.save_dir):
        # If it does not exist, create it
        os.makedirs(cfg.logger.save_dir)
        print(f"Directory created: {cfg.logger.save_dir}")
    else:
        print(f"Directory already exists: {cfg.logger.save_dir}")
    logger = pl.loggers.WandbLogger(
        config=dict(cfg),
        **cfg.logger,
    )
    cb = [
        callbacks.LearningRateMonitor(),
        # callbacks.EarlyStopping(**cfg.early_stopping),
        callbacks.ModelCheckpoint(**cfg.model_checkpoint),
        PredictionWriter(dataloaders=["train", "val", "test"]),
    ]
    dm = BaselineDataModule(
        **cfg.sampler, **cfg.data, cutoff=cfg.lightning_model.painn_kwargs.cutoff_dist
    )
    model = PaiNNforQM9(**cfg.lightning_model)
    trainer = Trainer(
        callbacks=cb,
        logger=logger,
        **cfg.trainer,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")
    trainer.predict(
        model,
        dataloaders=[
            dm.train_dataloader(shuffle=False),
            dm.val_dataloader(),
            dm.test_dataloader(),
        ],
        return_predictions=False,
        ckpt_path="best",
    )


if __name__ == "__main__":
    main()
