# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import math
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from iclr2018 import ScaleHyperprior
from config import cfg
from modules import Conv2d, ConvTranspose2d,activation_encoder
from runx.logx import logx

conf_acceleration = cfg['acceleration']
conf_nm_sparse = cfg['nm_sparse']
conf_compress_activation = cfg['compress_activation']
conf_sched = cfg['lr_schedule']
conf_train = cfg['trainer']

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["psnr"] = 10 * (torch.log(1. / out["mse_loss"])) / np.log(10)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, cfg):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    # assert len(inter_params) == 0
    # assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=conf_sched['lr'],
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=conf_sched['aux_lr'],
    )

    return optimizer, aux_optimizer

def train_one_epoch(
    model,  criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, batch_size
):
    model.train()
    device = next(model.parameters()).device

    if conf_compress_activation['adaptive_bit_depth_allocation']:
        tau = max(1 - epoch/50, 0.4)
        for m in model.modules():
            if hasattr(m, '_update_tau'):
                m._update_tau(tau)
                cur_tau = m.tau

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out_net = model(d)
        net_bytes = act_bytes = 0
        loss_w_bits = loss_a_bits = num_elems = 0
        #optimization for R-D 
        out_criterion = criterion(out_net, d)
        loss = out_criterion["loss"]
      
        #optimization for activation size
        if conf_compress_activation['if_compress_activation']:
            for m in model.modules():
                if isinstance(m, activation_encoder):
                        loss_a_bits += m.a_bits
            if conf_compress_activation['use_penalty']:
                loss+= conf_compress_activation['penalty']*loss_a_bits/(d.size(0)*d.size(2)*d.size(3))  
            act_bytes = loss_a_bits/8
                
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
    
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % ((len(train_dataloader.dataset) // batch_size) // 10) == 0:
            logx.msg(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f}|'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tPSNR: {out_criterion["psnr"].item():.2f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.4f}|'
                f"\tAux loss: {aux_loss.item():.2f}|"
                f"\tact_kbytes: {act_bytes/1000:.2f}"
            )


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    aux_loss = AverageMeter()
    ac_bytes =  loss_a_bits = act_bytes = num_elems  = 0

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)
            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(out_criterion['psnr'])
            if conf_compress_activation['if_compress_activation']:
                if conf_compress_activation['use_rans']:
                    for m in model.modules():
                        if isinstance(m, activation_encoder):
                            with open('bitstream.bin','ab') as f:
                                f.write(m.a_bits[0])      
                else:
                    for m in model.modules():        
                        if isinstance(m, activation_encoder):
                            loss_a_bits += m.a_bits
                    act_bytes = loss_a_bits/8      
    logx.msg(
        f"Test epoch {epoch}:"
        f"\tLoss: {loss.avg:.3f}| "
        f"\tMSE loss: {mse_loss.avg:.3f}|"
        f"\tPSNR: {psnr.avg:.2f}|"
        f"\tBpp loss: {bpp_loss.avg:.4f}|"
        f"\tAux loss: {aux_loss.avg:.2f}|"
        f"\tact_kbytes: {act_bytes/1000:.5f} \n"
    )

    return loss.avg

def save_checkpoint(state, is_best, filename):
    filename += 'checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-8]+"_best"+filename[-8:])


def main():
    logx.initialize(logdir=conf_train['log'], coolname=True, tensorboard=False)
    
    if conf_train['seed'] is not None:
        torch.manual_seed(conf_train['seed'])
        random.seed(conf_train['seed'])

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(conf_train['patch_size']), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [ transforms.ToTensor()]
    )

    train_dataset = ImageFolder(conf_train['dataset'], split=conf_train['train_split'], transform=train_transforms)
    test_dataset = ImageFolder(conf_train['dataset'], split=conf_train['test_split'], transform=test_transforms)

    # device = 'cuda' if conf_train['cuda'] and torch.cuda.is_available() else "cpu"
    device = torch.device('cuda',7) if conf_train['cuda'] and torch.cuda.is_available() else "cpu"
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=conf_train['batch_size'],
        num_workers=conf_train['num_workers'],
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=conf_train['test_batch_size'],
        num_workers=conf_train['num_workers'],
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # Create models
    net = ScaleHyperprior(N=128, M=192)
    
    net.set_compress_activation(conf_compress_activation)
    net = net.to(device)   


    # if conf_train['cuda'] and torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, cfg)
    criterion = RateDistortionLoss(lmbda=conf_train['lmbda'])

    checkpoint_path_existed = os.path.exists(conf_train['save_path'])
    if not checkpoint_path_existed:
        os.makedirs(conf_train['save_path'])
        print('===Checkpoint dir Made===')
    last_epoch = 0
    if conf_train['pretrained']:  # load from previous checkpoint
        print("Loading", conf_train['save_path'])
        checkpoint = torch.load(conf_train['save_path']+'checkpoint.pth.tar', map_location=device) 
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])

    best_loss = float("inf")
    for epoch in range(last_epoch, conf_train['epoches']):
        cur_lr = optimizer.param_groups[0]['lr']
        if epoch == 64:
            optimizer.param_groups[0]['lr'] = cur_lr*0.5
            cur_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        if epoch >= conf_nm_sparse['sparse_epoch'] :
            net.set_nm_sparse(conf_nm_sparse)
        if epoch >= conf_acceleration['acceleration_epoch'] :
            net.set_acceleration(conf_acceleration)


        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            conf_train['clip_max_norm'],
            conf_train['batch_size'],
        )

        print('======Test with Kodakset======')
        loss = test_epoch(epoch, test_dataloader, net, criterion)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint(
            {
                "epoch": epoch,
                "model": net.state_dict(),
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
            },
            is_best,
            conf_train['save_path'],
        )


if __name__ == "__main__":
    main()
