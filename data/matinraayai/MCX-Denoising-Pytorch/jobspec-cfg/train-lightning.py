import pytorch_lightning
import torch.nn
from torch.utils.data import DataLoader
from data.dataset import OsaDataset
from model.builder import Criterion, get_model
from solver.builder import build_optimizer, build_lr_scheduler
import argparse
from model.loss import SSIM, PSNR
import torch.nn as nn
from config import read_training_cfg_file
from data.augmentation import build_train_augmentor
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os


def get_args():
    parser = argparse.ArgumentParser(description="Denoising model training script")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml)')
    return parser.parse_args()


class TrainLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(get_model(**cfg.model))
        self.train_loss = Criterion(**cfg.loss).to(self.device)
        self.validation_losses = {'MSE': nn.MSELoss(),
                                  'SSIM': SSIM(**self.cfg.loss.ssim).to(self.device),
                                  'PSNR': PSNR()}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, x, y = batch
        y_hat = self.model(x)
        return self.train_loss(y, y_hat)

    def validation_step(self, batch, batch_idx):
        _, x, y = batch
        y_hat = self.model(x)
        loss_dict = {loss_key: loss_module(y, y_hat) for loss_key, loss_module in self.validation_losses.items()}
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = build_optimizer(self.cfg, self.model)
        lr_scheduler = build_lr_scheduler(self.cfg, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "MSE"}

    def train_dataloader(self):
        train_augmentor = build_train_augmentor(**self.cfg.aug)
        train_dataset = OsaDataset(self.cfg.dataset.train_path, self.cfg.dataset.input_labels,
                                   self.cfg.dataset.output_label, True, self.cfg.dataset.crop_size, train_augmentor)
        train_dataloader = DataLoader(train_dataset, self.cfg.solver.batch_size, shuffle=True,
                                      num_workers=self.cfg.dataset.dataloader_workers,
                                      pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        valid_dataset = OsaDataset(self.cfg.dataset.valid_path, self.cfg.dataset.valid_labels,
                                   self.cfg.dataset.output_label, True, None)
        valid_dataloader = DataLoader(valid_dataset, 1, shuffle=False,
                                      num_workers=self.cfg.dataset.dataloader_workers,
                                      pin_memory=True)
        return valid_dataloader


def main():
    # arguments
    args = get_args()
    print("Command line arguments:")
    print(args)
    # configurations
    cfg = read_training_cfg_file(args.config_file)
    print("Configuration details:")
    print(cfg)
    # Fix seed for determinism
    if cfg.pytorch_lightning.seed_everything:
        pytorch_lightning.seed_everything(cfg.pytorch_lightning.seed)
    module = TrainLightningModule(cfg)
    if not os.path.exists(cfg.pytorch_lightning.checkpoint_dir):
        os.makedirs(cfg.pytorch_lightning.checkpoint_dir, exist_ok=True)
    trainer = Trainer(default_root_dir=".",
                      resume_from_checkpoint=cfg.pytorch_lightning.resume_training_checkpoint,
                      gpus=cfg.pytorch_lightning.num_gpus,
                      num_nodes=cfg.pytorch_lightning.num_nodes, max_epochs=cfg.solver.total_iterations,
                      accelerator=cfg.pytorch_lightning.accelerator,
                      plugins=DDPPlugin(find_unused_parameters=False),
                      logger=TensorBoardLogger(save_dir=cfg.pytorch_lightning.checkpoint_dir,
                                               name=cfg.pytorch_lightning.experiment_name),
                      callbacks=[ModelCheckpoint(save_top_k=-1,
                                                 filename='{epoch:04d}-{MSE:.4f}-{SSIM:.4f}-{PSNR:.4f}')])
    trainer.fit(module)


if __name__ == '__main__':
    main()
