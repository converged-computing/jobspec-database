#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 01/06/2022 1:43 PM
# @Author : Ayush Chaturvedi
# @E-mail : ayushchatur@vt.edu
# @Site :
# @File : sparse_ddnet.py
# @Software: PyCharm
# from apex import amp
# import torch.cuda.nvtx as nvtx
from importlib.resources import read_text

import torch.nn.utils.prune as prune
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
import parser_util as prs
import os
from os import path
from PIL import Image

from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR,ReduceLROnPlateau,ExponentialLR

from socket import gethostname

def dd_train(args):
    port = args.port
    os.environ['MASTER_PORT'] = str(port)
    torch.manual_seed(torch.initial_seed())
    world_size =  int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node  = torch.cuda.device_count()
    if gpus_per_node >0:
        local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    else:
        local_rank = 0

    distback = args.distback
    dist.init_process_group(distback, rank=rank, world_size=world_size)
    print(f"Hello from local_rank: {local_rank} and global rank {dist.get_rank()} of world with size: {dist.get_world_size()} on {gethostname()} where there are {gpus_per_node} allocated GPUs per node.", flush=True)
    # torch.cuda.set_device(local_rank)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)
    if rank == 0: print(args)
    batch = args.batch
    # print(args)
    epochs = args.epochs
    retrain = args.retrain
    prune_t = args.prune_t
    prune_amt = float(args.prune_amt)
    # enable_gr = (args.enable_gr == "true")
    gr_mode = args.gr_mode
    mod = args.model
    gr_backend = args.gr_backend
    amp_enabled = (args.amp == "true")
    global dir_pre
    global gamma
    global beta
    dir_pre = args.out_dir
    num_w = args.num_w
    en_wan = args.wan
    inference = (args.do_infer == "true")
    enable_prof = (args.enable_profile == "true")
    new_load = (args.new_load == "true")
    gr_mode = (args.enable_gr == "true")
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu", local_rank)

    torch.cuda.manual_seed(1111)
    # necessary for AMP to work

    # model.to(device)
    model_file: str
    model_file = f'weights_{str(batch)}_{str(epochs + retrain)}.pt'
    # model_file = "/projects/synergy_lab/ayush/dense_weights_ddnet/weights_dense_" + str(epochs) + "_.pt"
    if mod == "vgg16":
        print("loading vgg16")
        from core.vgg16.ddnet_model import DD_net
        model = DD_net(devc=device)
        gamma = 0.03
        beta = 0.05
        # model_file = "/projects/synergy_lab/ayush/weights_vgg16/weights_dense_" + str(epochs) + ".pt"
    elif mod == "vgg19":
        print("loading vgg19")
        from core.vgg19.ddnet_model import DD_net
        model = DD_net(devc=device)
        gamma = 0.04
        beta = 0.04
        # model_file = "."
    else:
        print("loading vanilla ddnet")
        from core import DD_net
        model = DD_net()
        gamma = 0.1
        beta = 0.0
    model.to(device)

    if gr_mode:
        model = DDP(model, device_ids=[local_rank])
        model = torch.compile(model, fullgraph=True, mode=gr_mode, backend=gr_backend)

    else:
        model = DDP(model, device_ids=[local_rank])
    global learn_rate
    learn_rate = args.lr
    epsilon = 1e-8



    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, eps=epsilon)  #######ADAM CHANGE
    decayRate = args.dr
    sched_type = args.schedtype
    if sched_type == "cos" :
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=10,eta_min=0.0005)
    elif sched_type == "platu" :
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                               threshold=0.01, threshold_mode='rel',
                                                               cooldown=5, min_lr=0.005, eps=1e-03)
    else:
        scheduler = ExponentialLR(optimizer=optimizer, gamma=decayRate)


    # model_file = "1447477/weights_dense_" + str(epochs) + "_.pt"

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    # if en_wan > 0:
    #     wandb.watch(model, log_freq=100)
    dist.barrier()
    trainer = None
    if (not (path.exists(model_file))):
        print('model file not found')
    else:
        print(f'loading model file: {model_file}')
        model.load_state_dict(torch.load(model_file, map_location=map_location))

    print("initiating training")
    if mod == "ddnet":
        print('running ddnet')
        from core.sparse_ddnet_old_dl import SpraseDDnetOld
        trainer = SpraseDDnetOld(epochs, retrain, batch, model, optimizer, scheduler, world_size, prune_t,
                                 prune_amt, gamma, beta, dir_pre=dir_pre, amp=amp_enabled, sched_type=sched_type)
    else:
        print('running ddnet-ml-vgg')
        from core.sparse_ddnet_old_vgg import SpraseDDnetOld
        trainer = SpraseDDnetOld(epochs, retrain, batch, model, optimizer, scheduler, world_size, prune_t,
                             prune_amt, gamma, beta, dir_pre=dir_pre, amp=amp_enabled, sched_type=sched_type)
    trainer.train_ddnet(rank, local_rank, enable_profile=enable_prof)


    if rank == 0:
        print("saving model file")
        # model_file = f'weights_{str(batch)}_{str(epochs+retrain)}.pt'
        torch.save(model.state_dict(), dir_pre + "/" + model_file)
        if not inference:
            print("not doing inference.. training only script")
    # dist.barrier()
    dist.destroy_process_group()
    return


def main():

    parser = prs.get_parser()
    args = parser.parse_args()

    if(args.wan > 0):
        import wandb
        wandb.init()
    dd_train(args)


if __name__ == '__main__':
    main()
    exit()

