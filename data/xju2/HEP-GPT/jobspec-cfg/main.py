import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import hydra
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("sum", lambda x, y: x + y)
OmegaConf.register_new_resolver("gen_list", lambda x, y: [x] * y)

from typing import List, Tuple

import lightning as L
from lightning.pytorch.core import LightningDataModule, LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.callbacks import Callback

import torch

from src.utils import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def main(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains or Evaluation the model.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    torch.set_float32_matmul_precision("medium")

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed)

    stage = cfg.get("stage", "fit")

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if stage == "fit":
        log.info("Starting training!")
        trainer.fit(model=model,
                    train_dataloaders=datamodule.train_dataloader(),
                    val_dataloaders=datamodule.val_dataloader(),
                    ckpt_path=cfg.get("ckpt_path", None),
                    )
    elif stage == "test":
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")
    else:
        raise ValueError(f"Unknown stage: {stage}")


@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="main.yaml")
def lightning_main(cfg : DictConfig) -> None:
    main(cfg)

if __name__ == "__main__":
    lightning_main()
