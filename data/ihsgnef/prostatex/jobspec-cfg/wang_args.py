# -*- coding: utf-8 -*-
import os
import time
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from torchmetrics.functional.classification import auroc, stat_scores, average_precision, precision_recall_curve, auc
from pytorch_lightning.loggers import WandbLogger
import wandb

import warnings
warnings.filterwarnings("ignore")


'''
Selecting proper combination of mpMRI sequences for prostate cancer
classification using multi-input convolutional neuronal network
https://www.physicamedica.com/article/S1120-1797(20)30256-8/fulltext

'''


class MSDSC(pl.LightningModule):
    '''
    Multi-Scale Depthwise Separable Convolution Block
    '''

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


class WangClassifier(pl.LightningModule):

    def __init__(self, verbose=False, **config_kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.mri_sequences = self.hparams.mri_sequences
        self.num_sequences = len(self.hparams.mri_sequences)
        self.data_sequences = self.hparams.data_sequences
        self.mri_index = [self.data_sequences.find(s) for s in self.mri_sequences]
        self.fn_penalty = self.hparams.fn_penalty
        self.embed_dim = self.hparams.embed_dim

        self.register_buffer("w_ensemble", torch.tensor(
            [1 / (self.num_sequences + 1)] * (self.num_sequences + 1), requires_grad=False))

        out_size = 2 if self.hparams.pooling else 64
        self.conv = nn.ModuleList([
            nn.Sequential(*([MSDSC(16, self.hparams.pooling) for i in range(5)] + [nn.Flatten()]))
            for j in range(self.num_sequences)
        ])

        self.linear = nn.ModuleList([
            nn.Sequential(
                nn.Linear(16 * out_size ** 2, 64),
                nn.ReLU(),
                nn.Linear(64, self.embed_dim),
            ) for i in range(self.num_sequences)
        ])

        self.fusion = nn.Sequential(
            nn.Linear(16 * out_size ** 2 * self.num_sequences, 64),
            nn.ReLU(),
            nn.Linear(64, self.embed_dim),
        )

        if self.embed_dim > 1:
            self.classifier = nn.ModuleList([
                nn.Sequential(nn.ReLU(), nn.Linear(self.embed_dim, 1))
                for i in range(self.num_sequences + 1)
            ])

        if verbose:
            self.summarize()

    def criterion(self, logits, target):

        def seq_criterion(l, y, w, C=self.fn_penalty):
            p, lp, nlp = torch.sigmoid(l), nn.functional.logsigmoid(l), nn.functional.logsigmoid(-l)
            return -w * (C ** (1 - p) * y * lp + (1 - y) * nlp)

        seq_weight = torch.tensor([0.2] * self.num_sequences + [1], device=self.device)
        seq_loss = [seq_criterion(logits[:, i], target, seq_weight[i]).unsqueeze(1) for i in range(logits.shape[1])]
        loss = torch.cat(seq_loss, 1).sum(1).mean()
        return loss

    def metrics(self, prob, target, threshold=0.5):
        pred = (prob >= threshold).long()
        tp, fp, tn, fn, sup = stat_scores(pred, target, ignore_index=0)
        if 0 < sup < len(target):
            precision, recall, _ = precision_recall_curve(pred, target)
            auprc = auc(recall, precision)

        return {
            'pred': pred,
            'auc': auroc(prob, target) if 0 < sup < len(target) else None,
            'acc': (tp + tn) / (tp + tn + fp + fn),
            'tpr': tp / (tp + fn),
            'tnr': tn / (tn + fp),
            'ppv': tp / (tp + fp),
            'f1': 2 * tp / (2 * tp + fp + fn),
            'ap': average_precision(prob, target),
            'auprc': auprc if 0 < sup < len(target) else None,
        }

    def forward(self, x):
        conv = [
            conv(x[:, i].unsqueeze(1).repeat(1, 16, 1, 1))
            for i, conv in enumerate(self.conv)
        ]
        conv_x = torch.cat([c.unsqueeze(1) for c in conv], 1)
        linear = [linear(conv_x[:, i]) for i, linear in enumerate(self.linear)]

        if self.embed_dim > 1:
            fusion = self.fusion(torch.cat(conv, 1))
            x = linear + [fusion]
            x = [clf(x[i]) for i, clf in enumerate(self.classifier)]
            x = torch.cat(x, 1)
        else:
            linear_x = torch.cat(linear, 1)
            fusion_x = self.fusion(torch.cat(conv, 1))
            x = torch.cat((linear_x, fusion_x), 1)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x[:, self.mri_index]
        logits = self(x)
        y_hat = torch.sigmoid(logits)
        y_hat.require_grad = False
        loss = self.criterion(logits, y.float())
        m = self.metrics(y_hat.matmul(self.w_ensemble), y)
        # m = self.metrics(y_hat.mean(axis=1), y)
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_acc', m['acc'], prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x[:, self.mri_index]
        logits = self(x)
        y_hat = torch.sigmoid(logits)
        y_hat.require_grad = False
        loss = self.criterion(logits, y.float())
        ms = [self.metrics(y_hat[:, i], y) for i in range(y_hat.shape[1])]
        m = self.metrics(y_hat.matmul(self.w_ensemble), y)
        # m = self.metrics(y_hat.mean(axis=1), y)
        # print("W_en: " + ", ".join(map(lambda x: f"{x:.4f}", self.w_ensemble)))
        # print("Pred: " + "".join(map(str, m['pred'].tolist())))
        self.log('valid_loss', loss, sync_dist=True)
        self.log('valid_acc', m['acc'], prog_bar=True, sync_dist=True)
        self.log('valid_auc', m['auc'], prog_bar=True, sync_dist=True)
        self.log('valid_sensitivity', m['tpr'], sync_dist=True)
        self.log('valid_specificity', m['tnr'], sync_dist=True)
        self.log('valid_precision', m['ppv'], sync_dist=True)
        self.log('valid_f1', m['f1'], sync_dist=True)
        self.log('valid_ap', m['ap'], sync_dist=True)
        self.log('valid_auprc', m['auprc'], sync_dist=True)

        for i, m in enumerate(ms):
            sid = self.mri_sequences[i] if i < self.num_sequences else 'N'
            self.log(f'seq_{sid}_acc', m['acc'], sync_dist=True)
            self.log(f'seq_{sid}_auc', m['auc'], sync_dist=True)
            self.log(f'seq_{sid}_sensitivity', m['tpr'], sync_dist=True)
            self.log(f'seq_{sid}_specificity', m['tnr'], sync_dist=True)
            self.log(f'seq_{sid}_precision', m['ppv'], sync_dist=True)
            self.log(f'seq_{sid}_f1', m['f1'], sync_dist=True)
            self.log(f'seq_{sid}_ap', m['ap'], sync_dist=True)
            self.log(f'seq_{sid}_auprc', m['auprc'], sync_dist=True)

        s_bar = torch.tensor([m['auc'] for m in ms], device=self.device).mean()
        ss = torch.tensor([m['auc'] for m in ms], device=self.device)
        w_e = (ss.clamp(min=s_bar) - s_bar).type_as(ss)

        if w_e.sum() > 0:
            self.w_ensemble = w_e / w_e.sum()

        # w_e_columns = ["Step"] + list(self.mri_sequences + 'N')
        # w_e_data = [self.global_step] + [f"{w:.4f}" for w in self.w_ensemble]
        # wandb.log({"wEnsemble": wandb.Table(data=w_e_data, columns=w_e_columns)}, step=self.global_step)
        return {'valid_auc': m['auc']}

    def embed(self, x):
        x = x[:, self.mri_index]
        conv = [conv(x[:, i].unsqueeze(1).repeat(1, 16, 1, 1))
                for i, conv in enumerate(self.conv)]
        embeds = torch.cat(conv, 1)
        if self.embed_dim > 1:
            conv_x = torch.cat([c.unsqueeze(1) for c in conv], 1)
            linear = [linear(conv_x[:, i]) for i, linear in enumerate(self.linear)]
            linear_x = torch.cat(linear, 1)
            fusion_x = self.fusion(torch.cat(conv, 1))
            embeds = torch.cat((linear_x, fusion_x), 1)
        return embeds

    def predict(self, batch, multi=False, ensemble=True, prob=True, threshold=0.5):
        x = batch[:, self.mri_index]
        logits = self(x)
        y_hat = torch.sigmoid(logits)
        if multi:
            return y_hat
        pred = y_hat.matmul(self.w_ensemble) if ensemble else y_hat.mean(axis=1)
        if not prob:
            pred = (prob >= threshold).long()
        return pred

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
            self.hparams.train_dir,
            extensions='npy',
            loader=np.load,
            transform=self.parse_augmentation(),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.dataloader_num_workers,
            drop_last=True, shuffle=True)
        return dataloader

    def val_dataloader(self):
        dataset = torchvision.datasets.DatasetFolder(
            self.hparams.valid_dir,
            extensions='npy',
            loader=np.load,
            transform=transforms.ToTensor(),
        )
        batch_size = len(dataset) if self.hparams.eval_batch_size == -1 else self.hparams.eval_batch_size
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.hparams.dataloader_num_workers,
            drop_last=False,
            shuffle=False,
        )
        return dataloader

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--mri_sequences", default=None, type=str, required=True)
        parser.add_argument("--data_sequences", default=None, type=str, required=True)
        parser.add_argument("--pooling", action="store_true")
        parser.add_argument("--fn_penalty", default=20, type=int, help="Penalty for false negatives")
        parser.add_argument("--embed_dim", default=1, type=int, help="Embedding size")
        parser.add_argument("--horizontal_flip", default=0, type=float)
        parser.add_argument("--vertical_flip", default=0, type=float)
        parser.add_argument("--rotate", default=0, type=int)
        parser.add_argument("--translate", default=0, type=float)
        parser.add_argument("--scale", default=0, type=float)
        parser.add_argument("--shear", default=0, type=float)
        return parser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--loss", default="binary_cross_entropy", type=str)
    parser.add_argument("--metric", default="accuracy", type=str)
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--dataloader_num_workers", default=16, type=int)
    parser.add_argument("--train_dir", default=None, type=str, required=True)
    parser.add_argument("--valid_dir", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--wandb_group", default=None, type=str)
    parser.add_argument("--wandb_mode", default="online", type=str)
    parser.add_argument("--do_train", action="store_true")
    WangClassifier.add_model_specific_args(parser, os.getcwd())
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
    model = WangClassifier(**dict_args)

    # init model
    odir = Path(model.hparams.output_dir)
    odir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(os.path.join(model.hparams.output_dir, 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    # build logger
    experiment = wandb.init(
        mode=args.wandb_mode,
        group=args.wandb_group,
    )
    logger = WandbLogger(
        project="prostatex",
        experiment=experiment
    )

    # add custom checkpoints
    ckpt_path = os.path.join(
        args.output_dir, logger.version, "checkpoints",
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_path,
        filename="{epoch}-{valid_loss:.2f}",
        monitor="valid_loss",
        mode="min",
        save_last=True,
        save_top_k=3,
        verbose=True,
    )

    train_params = {}
    train_params["max_epochs"] = args.max_epochs
    if args.gpus == -1 or args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer.from_argparse_args(
        args,
        auto_select_gpus=True,
        weights_summary=None,
        callbacks=[checkpoint_callback],
        logger=logger,
        **train_params,
    )

    if args.do_train:
        trainer.fit(model)
        # save best model to `best_model.ckpt`
        target_path = os.path.join(ckpt_path, 'best_model.ckpt')
        print(f"Copy best model from {checkpoint_callback.best_model_path} to {target_path}.")
        shutil.copy(checkpoint_callback.best_model_path, target_path)


if __name__ == "__main__":
    main()
