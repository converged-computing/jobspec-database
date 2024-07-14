from __future__ import print_function, division
import wandb
import sys
import losses as Losses
sys.path.append('core')

import argparse
import os
#import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import experiment
import torch
import torch.nn as nn
import yaml
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import datasets
import core.sequence_handling_utils as seq_utils

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000
GB = 10**9
def sequence_loss(flow_preds, flow_gt, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model, epochs_sofar=0, last_step=-1):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], gamma=0.1, last_epoch=-1, verbose=False)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=args.num_steps, steps_per_epoch=943,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def save_model(epoch, model, optimizer, scheduler, val_loss_dict, loss_epoch, 
               img_error_epoch, tmp_error_epoch, experiment_dir, PATH):
    torch.save({'epoch' : epoch,
                'model' : model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'validation_loss' : val_loss_dict["Total"],
                'validation_img_error' : val_loss_dict["Img Error"],
                'validation_tmp_error' : val_loss_dict["Tmp Error"],
                'train_loss' : loss_epoch,
                'train_img_error' : img_error_epoch,
                'train_tmp_error' : tmp_error_epoch},
                     f'{experiment_dir}/{PATH}')
    
def train(args):
    wandb.init(project="test-project", entity="manalteam")
    
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    epoch = 0
    last_epoch = config.num_steps - 1
    
    cuda_to_use = "cuda:" + str(config.gpus[0])
    model.to(cuda_to_use)

    if config.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(config)
    optimizer, scheduler = fetch_optimizer(config, model)

    scaler = GradScaler(enabled=config.mixed_precision)

    VAL_FREQ = 10
    add_noise = False
    
    if config.restore_ckpt is not None:
        print("Found Checkpoint: ", config.restore_ckpt)
        ckpt = torch.load(config.restore_ckpt)
        epoch = ckpt['epoch'] + 1
        model.module.load_state_dict(ckpt['model'], strict=True)
        last_step_sched = 943 * epoch
        optimizer, scheduler = fetch_optimizer(config, model, epochs_sofar=epoch, last_step=last_step_sched)
        scheduler.last_epoch = last_step_sched
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        last_epoch = epoch + config.num_steps - 1
        print("I am loading an existing model from", config.restore_ckpt, 
              "at epoch", ckpt['epoch'], "so we are starting at epoch", epoch, 
              "with a training loss", ckpt['train_loss'], "and validation loss", ckpt['validation_loss'])
        
    model.train()
    
    experiment_dir = 'october_checkpoints/%s'% (config.name)
    os.mkdir(experiment_dir)
    print("I created the exp dir", experiment_dir)
    
    should_keep_training = True
    while should_keep_training:
        loss_epoch, tmp_error_epoch, img_error_epoch, spa_epoch, temp_epoch = 0, 0, 0, 0, 0
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            if (config.learn_temp):
                image_batch, patient_slice_id_batch = [x for x in data_blob] #  [B,C,H,W] new [B,N,H,W], [B,N,H,W]
                template_batch = None
            else:
                image_batch, template_batch, patient_slice_id_batch = [x for x in data_blob] #[B,C,H,W] new [B,N,H,W], [B,N,H,W]
                image_batch, template_batch = image_batch.to(cuda_to_use), template_batch.to(cuda_to_use)
            
            if config.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image_batch = (image_batch + stdv * torch.randn(*image_batch.shape).to(cuda_to_use)).clamp(0.0, 255.0)
                template_batch = (template_batch + stdv * torch.randn(*template_batch.shape).to(cuda_to_use)).clamp(0.0, 255.0)
            flow_predictions1, flow_predictions2, template_batch = model(image_batch, template_batch, iters=config.iters)

            # list of flow estimations with length iters, and each item of the list is [B, 2, H, W]   new [B, N, 2, H, W]  
            image_batch, template_batch = image_batch.to(cuda_to_use), template_batch.to(cuda_to_use)
            batch_loss_dict = Losses.disimilarity_loss(image_batch, template_batch, patient_slice_id_batch,
                                                              flow_predictions1, flow_predictions2, 
                                                              epoch=epoch, mode="training", 
                                                              i_batch=i_batch, args=config) 
            scaler.scale(batch_loss_dict["Total"]).backward()
           
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            
            scaler.update()
            loss_epoch += batch_loss_dict["Total"].item() / len(train_loader)
            spa_epoch += batch_loss_dict["Spatial"].item() / len(train_loader)
            temp_epoch += batch_loss_dict["Temporal"].item() / len(train_loader)
            if (config.model == 'group'):
                img_error_epoch += batch_loss_dict["Img Error"].item() / len(train_loader)
            tmp_error_epoch += batch_loss_dict["Temp Error"].item() / len(train_loader)
            
        wandb.log({"Training Total Loss": loss_epoch})
        wandb.log({"Training Img Error": img_error_epoch})
        wandb.log({"Training Tmp Error": tmp_error_epoch})
        wandb.log({"Training Spatial Loss": spa_epoch})
        wandb.log({"Training Temporal Loss": temp_epoch})
        
        print("Epoch:", epoch, "training loss", loss_epoch)
        print("Epoch:", epoch, "img training error", img_error_epoch)
        print("Epoch:", epoch, "tmp training error", tmp_error_epoch)
        # VALIDATION
        results = {}
        val_loss_dict = {}
        for val_dataset in config.validation:
            print("Im in validation", val_dataset)
            if val_dataset == 'chairs':
                results.update(evaluate.validate_chairs(model.module))
            elif val_dataset == 'sintel':
                results.update(evaluate.validate_sintel(model.module))
            elif val_dataset == 'kitti':
                results.update(evaluate.validate_kitti(model.module))
            elif val_dataset == 'acdc':
                
                val_loss_dict = evaluate.validate_acdc(model.module, config, epoch=epoch, mode='validation')

            wandb.log({"Validation Total Loss": val_loss_dict["Total"]})
            wandb.log({"Img Validation Error": val_loss_dict["Img Error"]})
            wandb.log({"Tmp Validation Error": val_loss_dict["Tmp Error"]})

        # Log every epoch
        if (epoch % 2 == 0):
            PATH = '%s_%d.pth' % (config.name, epoch)
            print("Should save", PATH)
            save_model(epoch, model, optimizer, scheduler, val_loss_dict, loss_epoch, 
                       img_error_epoch, tmp_error_epoch, experiment_dir, PATH)

        model.train()
        if config.stage != 'chairs':
            model.module.freeze_bn()
    
        epoch += 1 # Num of epochs

        if epoch > last_epoch:
            should_keep_training = False
            break

    
    model_dir = 'mymodels/%s'% (config.name)
    os.mkdir(model_dir)
    print("I created model dir", model_dir)
    PATH =  '%s_%d.pth' % (config.name, epoch-1)
    save_model(epoch-1, model, optimizer, scheduler, val_loss_dict, loss_epoch, 
               img_error_epoch, tmp_error_epoch, model_dir, PATH)

    return PATH


if __name__ == '__main__':
    with open("config.yml", "r") as stream:
        try:
            d = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='GroupwiseFull', help="name of experiment from config.yml")
    args = parser.parse_args()
    config = experiment.Experiment(d[args.experiment])    

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--dataset_folder', type=str)
    parser.add_argument('--max_seq_len', type=int, default=35)
    parser.add_argument('--add_normalisation', action='store_true')
    parser.add_argument('--beta_photo', type=float, default=1.0)
    parser.add_argument('--beta_spatial', type=float, default=0.0)
    parser.add_argument('--beta_temporal', type=float, default=0.0)
    
    
    args = parser.parse_args()
    '''
    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(config)