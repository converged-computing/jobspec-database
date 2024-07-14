# -*- coding: utf-8 -*-
import os
import time
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from torchmetrics.functional.classification import auroc, stat_scores, average_precision, precision_recall_curve, auc
from pytorch_lightning.loggers import WandbLogger
import wandb

import warnings
warnings.filterwarnings("ignore")


class MSDSC(pl.LightningModule):
    def __init__(self, in_channels, pooling=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels // 2, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels // 2, 1)
        )
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.MaxPool2d(2) if pooling else nn.Identity()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat((x1, x2), 1)
        x = self.layer(x)
        return x

class DWAC(pl.LightningModule):

    def __init__(self, verbose=False, **config_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.mri_sequences = self.hparams.mri_sequences
        self.num_sequences = len(self.hparams.mri_sequences)
        self.data_sequences = self.hparams.data_sequences
        self.mri_index = [self.data_sequences.find(s) for s in self.mri_sequences]
        
        self.embed_dim = self.hparams.embed_dim
        self.eps = self.hparams.eps
        self.gamma = self.hparams.gamma
        self.criterion = nn.NLLLoss(size_average=False)

        out_size = 2 if self.hparams.pooling else 64
        self.conv = nn.ModuleList([nn.Sequential(
            *([MSDSC(16, self.hparams.pooling) for i in range(5)] + [nn.Flatten()])) for j in range(self.num_sequences)])
        self.linear = nn.ModuleList([nn.Sequential(nn.Linear(
            16 * out_size**2, 64), nn.ReLU(), nn.Linear(64, self.embed_dim)) for i in range(self.num_sequences)])
        self.fusion = nn.Sequential(
            nn.Linear(16 * out_size**2 * self.num_sequences, 64), nn.ReLU(), nn.Linear(64, self.embed_dim))
        if self.hparams.merge_seq:
            self.merge_dim = self.hparams.merge_dim 
            self.merger = nn.Sequential(nn.ReLU(), nn.Linear(self.embed_dim * (self.num_sequences+1), self.merge_dim))
        else: 
            self.merger = nn.Identity()

        if self.hparams.kernel == 'laplace':
            print("Using Laplace kernel")
            self.distance_metric = self._laplacian_kernel
        elif self.hparams.kernel == 'invquad':
            print("Using Inverse Quadratic kernel with smoothing parameter {:.3f}".format(self.gamma))
            self.distance_metric = self._inverse_quadratic
        elif self.hparams.kernel == 'gaussian':
            print("Using Guassian kernel")
            self.distance_metric = self._gaussian_kernel
        elif self.hparams.kernel == 'sigmoid':
            print("Using Sigmoid kernel")
            self.distance_metric = self._sigmoid
        elif self.hparams.kernel == 'softplus':
            print("Using Softplus kernel")
            self.distance_metric = self._softplus
        elif self.hparams.kernel == 'relu':
            print("Using ReLU kernel")
            self.distance_metric = self._relu
        else:
            raise ValueError('Invalid Kernel')

        if verbose: 
            self.summarize()


    def metrics(self, prob, target, threshold=0.5):
        pred = (prob >= threshold).long()
        tp, fp, tn, fn, sup = stat_scores(pred, target, ignore_index=0)
        if 0 < sup < len(target):
            precision, recall, _ = precision_recall_curve(pred, target)
            auprc = auc(recall, precision)
        m = {}
        m['pred'] = pred
        m['auc'] = auroc(prob, target) if 0 < sup < len(target) else None
        m['acc'] = (tp + tn) / (tp + tn + fp + fn)
        m['tpr'] = tp / (tp + fn)
        m['tnr'] = tn / (tn + fp)
        m['ppv'] = tp / (tp + fp)
        m['f1'] = 2 * tp / (2 * tp + fp + fn)
        m['ap'] = average_precision(prob, target)
        m['auprc'] = auprc if 0 < sup < len(target) else None
        return m

    def embed(self, x):
        x = x[:, self.mri_index]
        conv = [conv(x[:, i].unsqueeze(1).repeat(1, 16, 1, 1))
                for i, conv in enumerate(self.conv)]
        embeds = torch.cat(conv, 1)
        if self.embed_dim > 1:
            conv_x = torch.cat([c.unsqueeze(1) for c in conv], 1)
            linear = [linear(conv_x[:, i]) for i, linear in enumerate(self.linear)]
            fusion = self.fusion(torch.cat(conv, 1))
            embeds = torch.cat(linear + [fusion], 1)
        else:
            linear = torch.cat(linear, 1)
            fusion = self.fusion(torch.cat(conv, 1))
            embeds = torch.cat((linear, fusion), 1)
        embeds = self.merger(embeds)
        return embeds

    def forward(self, x, y):
        z = self.embed(x)
        norm = z.pow(2).sum(dim=1)
        dists = torch.mm(z, z.t()).mul(-2).add(norm).t().add(norm).t()
        dists = self.distance_metric(dists)
        dists = dists.mul((1 != torch.eye(z.shape[0], device=z.device)).float())
        class_mask = torch.zeros(z.shape[0], 
                                 2, # self.n_classes
                                 device=z.device)
        class_mask.scatter_(1, y.view(z.shape[0], 1), 1)
        class_dists = torch.mm(dists, class_mask).add(self.eps)  # [batch_size, n_classes]
        probs = torch.div(class_dists.t(), class_dists.sum(dim=1)).log().t()

        total_loss = self.criterion(probs, y)
        output_dict = {
                'probs': probs,
                'loss': total_loss.div(x.shape[0]),
                'total_loss': total_loss,
                }
        return output_dict

    def training_step(self, batch, batch_idx):
        x, y = batch
        output_dict = self(x, y)
        m = self.metrics(output_dict['probs'][:, 1].exp(), y)
        self.log('train_loss', output_dict['loss'], sync_dist=True)
        self.log('train_acc', m['acc'], prog_bar=True, sync_dist=True)
        return output_dict['loss']

    def classify(self, z, z_norm, ref_z, ref_y):
        ref_norm = ref_z.pow(2).sum(dim=1)
        dists = torch.mm(z, ref_z.t()).mul(-2).add(ref_norm).t().add(z_norm).t()
        dists = self.distance_metric(dists)

        class_mask = torch.zeros(ref_z.shape[0], 
                                 2, # self.n_classes,
                                 device=ref_z.device)
        class_mask.scatter_(1, ref_y.view(ref_z.shape[0], 1), 1)
        class_dists = torch.mm(dists, class_mask)

        output_dict = {
                'class_dists': class_dists,
                }
        return output_dict

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.embed(x)
        z_norm = z.pow(2).sum(dim=1)
        batch = list(iter(self.ref_dataloader()))
        ref_x = batch[0][0].to(z.device)
        ref_y = batch[0][1].to(z.device)
        ref_z = self.embed(ref_x)
        class_dists = self.classify(z, z_norm, ref_z, ref_y)['class_dists']

        probs = class_dists.div(class_dists.sum(dim=1, keepdim=True)).log()
        total_loss = self.criterion(probs, y)
        loss = total_loss.div(x.shape[0])
        m = self.metrics(probs[:, 1].exp(), y)
        self.log('valid_loss', loss, sync_dist=True)
        self.log('valid_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('valid_auc', m['auc'], prog_bar=True, sync_dist=True)
        self.log('valid_sensitivity', m['tpr'], sync_dist=True)
        self.log('valid_specificity', m['tnr'], sync_dist=True)
        self.log('valid_precision', m['ppv'], sync_dist=True)
        self.log('valid_f1', m['f1'], sync_dist=True)
        self.log('valid_ap', m['ap'], sync_dist=True)
        self.log('valid_auprc', m['auprc'], sync_dist=True)
        return {'valid_loss': loss, 'valid_auc': m['auc']}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    
    def parse_augmentation(self):
        affine = {}
        affine["degrees"] = self.hparams.rotate
        if self.hparams.translate > 0: 
            translate = self.hparams.translate
            affine["translate"] = (translate, translate)
        if self.hparams.scale > 0: 
            scale = self.hparams.scale
            affine["scale"] = (1 - scale, 1 + scale)
        if self.hparams.shear > 0:
            shear = self.hparams.shear
            affine["shear"] = (-shear, shear, -shear, shear)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(self.hparams.horizontal_flip),
            transforms.RandomVerticalFlip(self.hparams.vertical_flip),
            transforms.RandomAffine(**affine)
        ])
        return transform

    def train_dataloader(self):
        dataset = torchvision.datasets.DatasetFolder(
            self.hparams.train_dir, extensions='npy', loader=np.load, transform=self.parse_augmentation()
            )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.hparams.train_batch_size, 
            num_workers=self.hparams.dataloader_num_workers, 
            drop_last=True, shuffle=True)
        return dataloader

    def val_dataloader(self):
        dataset = torchvision.datasets.DatasetFolder(
            self.hparams.valid_dir, extensions='npy', loader=np.load, transform=transforms.ToTensor()
            )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=len(dataset), 
            num_workers=self.hparams.dataloader_num_workers, 
            drop_last=False, shuffle=False)
        return dataloader

    def ref_dataloader(self):
        dataset = torchvision.datasets.DatasetFolder(
            self.hparams.train_dir, extensions='npy', loader=np.load, transform=transforms.ToTensor()
            )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=len(dataset),
            num_workers=self.hparams.dataloader_num_workers, 
            drop_last=False, shuffle=False)
        return dataloader

    def _gaussian_kernel(self, dists):
        return dists.mul_(-1 * self.gamma).exp_()

    def _laplacian_kernel(self, dists):
        return dists.pow_(0.5).mul_(-0.5 * self.gamma).exp_()

    def _inverse_quadratic(self, dists):
        return 1.0 / (self.gamma + dists)

    def _sigmoid(self, dists):
        return F.sigmoid(self.gamma - dists)

    def _softplus(self, dists):
        return F.softplus(self.gamma - dists)

    def _relu(self, dists):
        return F.relu(self.gamma - dists)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--mri_sequences", default=None, type=str, required=True)
        parser.add_argument("--data_sequences", default=None, type=str, required=True)
        parser.add_argument("--pooling", action="store_true")
        parser.add_argument("--merge_seq", action="store_true")
        parser.add_argument("--embed_dim", default=10, type=int, help="Embedding size")
        parser.add_argument("--merge_dim", default=10, type=int, help="Embedding size")
        parser.add_argument('--kernel', type=str, default='gaussian', help='hparam for kernel [guassian|laplace|invquad]')
        parser.add_argument('--gamma', type=float, default=1.0, help='hparam for kernel')
        parser.add_argument('--eps', type=float, default=1e-12, help='label smoothing factor for learning')
        parser.add_argument("--horizontal_flip", default=0, type=float)
        parser.add_argument("--vertical_flip", default=0, type=float)
        parser.add_argument("--rotate", default=0, type=int)
        parser.add_argument("--translate", default=0, type=float)
        parser.add_argument("--scale", default=0, type=float)
        parser.add_argument("--shear", default=0, type=float)
        return parser

def train(
    model:DWAC,
    args: argparse.Namespace,
    early_stopping_callback=False,
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs
    ):

    # init model
    odir = Path(model.hparams.output_dir)
    odir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(os.path.join(model.hparams.output_dir, 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    # build logger
    ## WandB logger
    experiment = wandb.init(
        mode=args.wandb_mode, 
        group=args.wandb_group
    )
    logger = WandbLogger(
        project="prostatex",
        experiment=experiment
    )

    # add custom checkpoints
    ckpt_path = os.path.join(
        args.output_dir, logger.version, "checkpoints",
    )
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_path, filename="{epoch}-{valid_loss:.2f}", monitor="valid_loss", mode="min", save_last=True, save_top_k=3, verbose=True
        )

    train_params = {}
    train_params["max_epochs"] = args.max_epochs
    if args.gpus == -1 or args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer.from_argparse_args(
        args,
        auto_select_gpus=True,
        weights_summary=None,
        callbacks=extra_callbacks + [checkpoint_callback],
        logger=logger,
        **train_params,
    )

    if args.do_train:
        trainer.fit(model)
        # save best model to `best_model.ckpt`
        target_path = os.path.join(ckpt_path, 'best_model.ckpt')
        print(f"Copy best model from {checkpoint_callback.best_model_path} to {target_path}.")
        shutil.copy(checkpoint_callback.best_model_path, target_path)

    return trainer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--dataloader_num_workers", default=16, type=int)
    parser.add_argument("--train_dir", default=None, type=str, required=True)
    parser.add_argument("--valid_dir", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--wandb_group", default=None, type=str)
    parser.add_argument("--wandb_mode", default="online", type=str)
    parser.add_argument("--do_train", action="store_true")
    DWAC.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    print(args)

    pl.seed_everything(args.seed)

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "./results",
            f"{__name__}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(args.output_dir)
    
    dict_args = vars(args)
    model = DWAC(**dict_args)
    trainer = train(model, args)

if __name__ == "__main__":
    main()
