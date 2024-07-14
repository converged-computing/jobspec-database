import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

import wandb
import torch
from lightning.pytorch.loggers import WandbLogger

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("len", lambda x: len(x))


@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    dcfg = OmegaConf.to_container(cfg, resolve=True)

    dm = instantiate(cfg.dataset)
    model = instantiate(cfg.model) # , datamodule=dm)
    model = model.to(memory_format=torch.channels_last)

    # Compared to WandbLogger(config=hcfg),
    # these two steps also work with wandb sweeps
    logger = WandbLogger(reinit=True)
    logger.experiment.config.setdefaults(dcfg)
    callbacks = [instantiate(cb_cfg) for _, cb_cfg in cfg.callbacks.items()]

    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(model, dm)
    # Since we specify num_steps, recalculate val and test metrics at the end
    trainer.validate(model, dm)
    trainer.test(model, dm)

    wandb.finish()  # required for multi-runs


if __name__ == "__main__":
    main()
