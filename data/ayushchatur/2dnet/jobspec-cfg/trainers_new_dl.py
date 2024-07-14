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

from ddnet_utils.pruning import mag_prune,ln_struc_spar,unstructured_sparsity
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR,ReduceLROnPlateau,ExponentialLR
from core import MSSSIM, SSIM,VGGloss
from ddnet_utils import serialize_loss_item
from socket import gethostname
import torch.cuda.amp as amp

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
    global batch_size
    batch_size = args.batch
    # print(args)
    global epochs
    epochs = args.epochs
    global retrain
    retrain = args.retrain
    global prune_t
    prune_t = args.prune_t
    global prune_amt
    prune_amt = float(args.prune_amt)

    global amp_enabled
    amp_enabled = (args.amp == "true")
    global dir_pre
    dir_pre = args.out_dir
    global gamma
    global beta
    mod = args.model
    enable_prof = (args.enable_profile == "true")

    gr_mode = (args.enable_gr == "true")
    gr_backend = args.gr_backend

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu", local_rank)


    torch.cuda.manual_seed(1111)
    # necessary for AMP to work
    model_file: str
    model_file = f'weights_{str(batch_size)}_{str(epochs + retrain)}.pt'
    if mod == "vgg16":
        from core.vgg16.ddnet_model import DD_net
        model = DD_net(devc=device)
        gamma = 0.03
        beta = 0.05
        # model_file = "/projects/synergy_lab/ayush/weights_vgg16/weights_dense_" + str(
        #     epochs) + ".pt"
    elif mod == "vgg19":
        from core.vgg19.ddnet_model import DD_net
        model = DD_net(devc=device)
        gamma = 0.04
        beta = 0.04
        # model_file = "/projects/synergy_lab/ayush/weights_vgg19/weights_dense_" + str(
        #     epochs) + ".pt"
    else:
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

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    # if en_wan > 0:
    #     wandb.watch(model, log_freq=100)

    # dist.barrier()
    try:

        if (not (path.exists(model_file))):
            print('model file not found')
        else:
            print(f'loading model file: {model_file}')
            model.load_state_dict(torch.load(model_file, map_location=map_location))
    except FileExistsError as f:
        print(f'file not found {f}')
    except Exception as e:
        print(f'error loading model: {e}')

    if mod =="ddnet":
        print("training ddnet")
        trainer_new(model, world_size, rank, local_rank, scheduler, optimizer, sched_type, dir_pre, enable_prof )
    else:
        print("training ddnet-ml-vgg")
        trainer_new_vgg(model, world_size, rank, local_rank, scheduler, optimizer, sched_type, dir_pre, enable_prof)
    if rank == 0:
        print("saving model file")
        torch.save(model.state_dict(), dir_pre + "/" + model_file)
    print("not doing inference.. training only script")
    # dist.barrier()
    dist.destroy_process_group()
    return

def trainer_new(model, world_size, global_rank, local_rank,scheduler, optimizer, sched_type = "expo" , output_path=".", enable_profile=False):
    global dir_pre
    dir_pre = output_path
    print(f"staging dataset on GPU {local_rank} of node {gethostname()}")
    root_train_h = "/projects/synergy_lab/garvit217/enhancement_data/train/HQ/"
    root_train_l = "/projects/synergy_lab/garvit217/enhancement_data/train/LQ/"
    root_val_h = "/projects/synergy_lab/garvit217/enhancement_data/val/HQ/"
    root_val_l = "/projects/synergy_lab/garvit217/enhancement_data/val/LQ/"
    from data_loader.custom_load import CTDataset
    train_loader = CTDataset(root_train_h, root_train_l, 5120, local_rank,  batch_size)
    val_loader = CTDataset(root_val_h, root_val_l, 784, local_rank,  batch_size)
    scaler = torch.cuda.amp.GradScaler()
    sparsified = False
    densetime = 0
    # list of random indexes
    g = torch.Generator()
    g.manual_seed(0)

    from ddnet_utils import init_loss_params
    train_total_loss, train_MSSSIM_loss, train_MSE_loss, val_total_loss, val_MSSSIM_loss, val_MSE_loss = init_loss_params()

        #         train_sampler.set_epoch(epochs + prune_ep)
    print(f"beginning training epochs on rank: {global_rank} ")
    print(f'profiling: {enable_profile}')
    q_fact_train = len(train_loader) // world_size
    q_fact_val = len(val_loader) // world_size
    if global_rank == 0: print(f"q_factor train {q_fact_train} , qfactor va : {q_fact_val} ")
    start = datetime.now()

    for k in range( epochs +  retrain):
        print(f"epoch: {k}")
        if global_rank == 0:
            train_index_list = torch.randperm(len(train_loader), generator=g).tolist()
            val_index_list = torch.randperm(len(val_loader), generator=g).tolist()
        else:
            train_index_list = [0 for i in range(len(train_loader))]
            val_index_list = [0 for i in range(len(val_loader))]
        # share permuted list of index with all ranks
        dist.broadcast_object_list(train_index_list, src=0)
        dist.broadcast_object_list(val_index_list, src=0)
        # dist.barrier()
        #
        # train_items_per_rank = math.ceil((len( train_loader) -  world_size) /  world_size)
        # val_items_per_rank = math.ceil((len( train_loader) -  world_size) /  world_size)
        train_index_list = train_index_list[global_rank * q_fact_train: (global_rank * q_fact_train + q_fact_train)]
        val_index_list = val_index_list[global_rank * q_fact_val: (global_rank * q_fact_val + q_fact_val)]
        train_index_list = [int(x) for x in train_index_list]
        val_index_list = [int(x) for x in val_index_list]

        # print(f"rank {global_rank} index list: {train_index_list}")
        # train_index_list = [list(train_index_list[i:i +  batch_size]) for i in
        #                     range(0, len(train_index_list),  batch_size)]
        #
        # val_index_list = [list(val_index_list[i:i +  batch_size]) for i in range(0, len(val_index_list),  batch_size)]
        # print(f"rank {global_rank} val_index_list list: {val_index_list}")
        if enable_profile:
            import nvidia_dlprof_pytorch_nvtx
            nvidia_dlprof_pytorch_nvtx.init(enable_function_stack=True)
            train_total, train_mse, train_msi, val_total, val_mse, val_msi = \
                 _epoch_profile(model, train_index_list, val_index_list, train_loader, val_loader, scheduler, optimizer, scaler, sched_type)
            train_total_loss[k] = train_total
            train_MSE_loss[k] = train_mse
            train_MSSSIM_loss[k] = train_msi
            val_total_loss[k] = val_total
            val_MSE_loss[k] = val_mse
            val_MSSSIM_loss[k] = val_msi
        else:
            print(f'length of training indices: {len(train_index_list)} for rank: {global_rank}')
            for index in range(0, len(train_index_list), batch_size):
                # print(f"fetching first { batch_size} items from index: {index}: ")
                # print(f"fetching indices: {train_index_list[index: index+  batch_size]}")
                sample_batched = train_loader.get_item(train_index_list[index: index + batch_size])
                # print(f"item recieved:  {sample_batched}")
                HQ_img, LQ_img, maxs, mins, file_name = sample_batched['HQ'], sample_batched['LQ'], \
                    sample_batched['max'], sample_batched['min'], sample_batched['vol']
                # print('indexes: ', idx)
                # print('shape: ', HQ_img.shape)
                # print('device: ', HQ_img.get_device())
                # print("got items")
                targets = HQ_img
                inputs = LQ_img
                with amp.autocast(enabled=amp_enabled):
                    outputs = model(inputs)
                    MSE_loss = nn.MSELoss()(outputs, targets)
                    MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
                    loss = MSE_loss + 0.1 * (MSSSIM_loss)
                    # print(loss)
                # print('calculating backpass')
                train_MSE_loss[k].append(MSE_loss.item())
                train_MSSSIM_loss[k].append(MSSSIM_loss.item())
                train_total_loss[k].append(loss.item())
                # model.zero_grad()
                optimizer.zero_grad(set_to_none=True)
                # BW pass
                if amp_enabled:
                    # print('bw pass')
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    #                 print('loss_bacl')
                    loss.backward()
                    #                 print('optimia')
                    optimizer.step()
            print("schelud")
            if sched_type == "platu":
                scheduler.step(loss)
            else:
                scheduler.step()

            print("Validation")
            for index in range(0, len(val_index_list), batch_size):
                sample_batched = val_loader.get_item(val_index_list[index: index + batch_size])
                HQ_img, LQ_img, maxs, mins, fname = sample_batched['HQ'], sample_batched['LQ'], sample_batched['max'], \
                sample_batched['min'], sample_batched['vol']

                with amp.autocast(enabled=amp_enabled):
                    outputs = model(LQ_img)
                    MSE_loss = nn.MSELoss()(outputs, HQ_img)
                    MSSSIM_loss = 1 - MSSSIM()(outputs, HQ_img)
                    # loss = nn.MSELoss()(outputs , targets_val) + 0.1*(1-MSSSIM()(outputs,targets_val))
                    loss = MSE_loss + 0.1 * (MSSSIM_loss)

                val_MSE_loss[k].append(MSE_loss.item())
                val_total_loss[k].append(loss.item())
                val_MSSSIM_loss[k].append(MSSSIM_loss.item())
        # dist.barrier()

        if sparsified == False and  retrain > 0 and k == ( epochs - 1):
            densetime = str(datetime.now() - start)
            print('pruning model on epoch: ', k)
            if  prune_t == "mag":
                print("pruning model by top k with %: ",  prune_amt)
                mag_prune( model,  prune_amt)
            elif  prune_t == "l1_struc":
                print("pruning model by L1 structured with %: ",  prune_amt)
                ln_struc_spar( model,  prune_amt)
            else:
                print("pruning model by random unstructured with %: ",  prune_amt)
                unstructured_sparsity( model,  prune_amt)
            sparsified = True
    print("total time : ", str(datetime.now() - start), ' dense time: ', densetime)
    serialize_loss_item(dir_pre, "train_mse_loss", train_MSE_loss, global_rank)
    serialize_loss_item(dir_pre, "train_total_loss", train_total_loss, global_rank)
    serialize_loss_item(dir_pre, "train_mssim_loss", train_MSSSIM_loss, global_rank)
    serialize_loss_item(dir_pre, "val_mse_loss", val_MSE_loss, global_rank)
    serialize_loss_item(dir_pre, "val_total_loss", val_total_loss, global_rank)
    serialize_loss_item(dir_pre, "val_mssim_loss", val_MSSSIM_loss, global_rank)

def trainer_new_vgg(model, world_size, global_rank, local_rank,scheduler, optimizer, sched_type = "expo" , output_path=".", enable_profile=False):
    global dir_pre
    dir_pre = output_path
    print(f"staging dataset on GPU {local_rank} of node {gethostname()}")
    root_train_h = "/projects/synergy_lab/garvit217/enhancement_data/train/HQ/"
    root_train_l = "/projects/synergy_lab/garvit217/enhancement_data/train/LQ/"
    root_val_h = "/projects/synergy_lab/garvit217/enhancement_data/val/HQ/"
    root_val_l = "/projects/synergy_lab/garvit217/enhancement_data/val/LQ/"
    from data_loader.custom_load import CTDataset
    train_loader = CTDataset(root_train_h, root_train_l, 5120, local_rank, batch_size)
    val_loader = CTDataset(root_val_h, root_val_l, 784, local_rank, batch_size)
    scaler = torch.cuda.amp.GradScaler()
    sparsified = False
    densetime = 0
    # list of random indexes
    g = torch.Generator()
    g.manual_seed(0)

    from ddnet_utils import init_vggloss_params
    train_total_loss, train_MSSSIM_loss, train_MSE_loss, train_vgg_loss, val_total_loss, val_MSSSIM_loss, val_MSE_loss, val_vgg_loss = init_vggloss_params()

    #         train_sampler.set_epoch(epochs + prune_ep)
    print(f"beginning training epochs on rank: {global_rank} ")
    print(f'profiling: {enable_profile}')

    q_fact_train = len(train_loader) // world_size
    q_fact_val = len(val_loader) // world_size
    if global_rank == 0: print(f"q_factor train {q_fact_train} , qfactor va : {q_fact_val} ")
    start = datetime.now()

    for k in range(1, epochs +  retrain + 1):
        print(f"epoch: {k}")
        if global_rank == 0:
            train_index_list = torch.randperm(len(train_loader), generator=g).tolist()
            val_index_list = torch.randperm(len(val_loader), generator=g).tolist()
        else:
            train_index_list = [0 for i in range(len(train_loader))]
            val_index_list = [0 for i in range(len(val_loader))]
        # share permuted list of index with all ranks
        dist.broadcast_object_list(train_index_list, src=0)
        dist.broadcast_object_list(val_index_list, src=0)
        # dist.barrier()
        #
        # train_items_per_rank = math.ceil((len( train_loader) -  world_size) /  world_size)
        # val_items_per_rank = math.ceil((len( train_loader) -  world_size) /  world_size)
        train_index_list = train_index_list[global_rank * q_fact_train: (global_rank * q_fact_train + q_fact_train)]
        val_index_list = val_index_list[global_rank * q_fact_val: (global_rank * q_fact_val + q_fact_val)]
        train_index_list = [int(x) for x in train_index_list]
        val_index_list = [int(x) for x in val_index_list]

        if enable_profile:
            import nvidia_dlprof_pytorch_nvtx
            nvidia_dlprof_pytorch_nvtx.init(enable_function_stack=True)
            _ =  _epoch_profile(model, train_index_list, val_index_list, train_loader, val_loader, scheduler, optimizer, scaler, sched_type)

        else:
            print(f'length of training indices: {len(train_index_list)} for rank: {global_rank}')
            for index in range(0, len(train_index_list), batch_size):
                # print(f"fetching first { batch_size} items from index: {index}: ")
                # print(f"fetching indices: {train_index_list[index: index+  batch_size]}")
                sample_batched = train_loader.get_item(train_index_list[index: index + batch_size])
                # print(f"item recieved:  {sample_batched}")
                HQ_img, LQ_img, maxs, mins, file_name = sample_batched['HQ'], sample_batched['LQ'], \
                    sample_batched['max'], sample_batched['min'], sample_batched['vol']
                # print('indexes: ', idx)
                # print('shape: ', HQ_img.shape)
                # print('device: ', HQ_img.get_device())
                # print("got items")

                with amp.autocast(enabled=amp_enabled):
                    outputs, out_b3, out_b1, tar_b3, tar_b1 = model(LQ_img, HQ_img)
                    MSE_loss = nn.MSELoss()(outputs, HQ_img)
                    MSSSIM_loss = 1 - MSSSIM()(outputs, HQ_img)
                    loss_vgg_b1 = VGGloss()(out_b1,
                                            tar_b1)  # enhanced image : [1, 256, 56, 56] dim should be same (1,256,56,56)
                    loss_vgg_b3 = VGGloss()(out_b3, tar_b3)
                    loss_vgg = (loss_vgg_b3 + loss_vgg_b1)
                    # loss = nn.MSELoss()(outputs , targets_val) + 0.1*(1-MSSSIM()(outputs,targets_val))
                    loss = MSE_loss + gamma * (MSSSIM_loss) + beta * loss_vgg
                    # print(loss)
                # print('calculating backpass')
                train_MSE_loss[k].append(MSE_loss.item())
                train_MSSSIM_loss[k].append(MSSSIM_loss.item())
                train_vgg_loss[k].append(loss_vgg.item())
                train_total_loss[k].append(loss.item())
                # model.zero_grad()
                optimizer.zero_grad(set_to_none=True)
                # BW pass
                if amp_enabled:
                    # print('bw pass')
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    #                 print('loss_bacl')
                    loss.backward()
                    #                 print('optimia')
                    optimizer.step()
            print("schelud")
            if sched_type == "platu":
                scheduler.step(loss)
            else:
                scheduler.step()

            print("Validation")
            for index in range(0, len(val_index_list), batch_size):
                sample_batched = val_loader.get_item(val_index_list[index: index + batch_size])
                HQ_img, LQ_img, maxs, mins, fname = sample_batched['HQ'], sample_batched['LQ'], sample_batched['max'], \
                    sample_batched['min'], sample_batched['vol']

                with amp.autocast(enabled=amp_enabled):
                    outputs, out_b3, out_b1, tar_b3, tar_b1 = model(LQ_img, HQ_img)
                    MSE_loss_val = nn.MSELoss()(outputs, HQ_img)
                    MSSSIM_loss_val = 1 - MSSSIM()(outputs, HQ_img)
                    # loss = nn.MSELoss()(outputs , targets_val) + 0.1*(1-MSSSIM()(outputs,targets_val))
                    loss_vgg_b1 = torch.mean(torch.abs(torch.sub(out_b3,
                                                                 tar_b3)))  # enhanced image : [1, 256, 56, 56] dim should be same (1,256,56,56)
                    loss_vgg_b3 = torch.mean(torch.abs(torch.sub(out_b1,
                                                                 tar_b1)))
                    val_vgg = (loss_vgg_b3 + loss_vgg_b1)
                    # loss = nn.MSELoss()(outputs , targets_val) + 0.1*(1-MSSSIM()(outputs,targets_val))
                    total_val = MSE_loss_val + gamma * (MSSSIM_loss_val) + beta * val_vgg

                val_MSE_loss[k].append(MSE_loss_val.item())
                val_total_loss[k].append(total_val.item())
                val_vgg_loss[k].append(val_vgg.item())
                val_MSSSIM_loss[k].append(MSSSIM_loss_val.item())


        # dist.barrier()

        if sparsified == False and  retrain > 0 and k == epochs :
            # start = datetime.now()
            densetime = str(datetime.now() - start)

            print(f'pruning model on epoch: {k}, dense time spent: {densetime} ')
            if  prune_t == "mag":
                print("pruning model by top k with %: ",  prune_amt)
                mag_prune( model,  prune_amt)
            elif  prune_t == "l1_struc":
                print("pruning model by L1 structured with %: ",  prune_amt)
                ln_struc_spar( model,  prune_amt)
            else:
                print("pruning model by random unstructured with %: ",  prune_amt)
                unstructured_sparsity( model,  prune_amt)
            sparsified = True
    print("total time : ", str(datetime.now() - start), ' dense time: ', densetime)
    serialize_loss_item(dir_pre, "train_mse_loss", train_MSE_loss, global_rank)
    serialize_loss_item(dir_pre, "train_total_loss", train_total_loss, global_rank)
    serialize_loss_item(dir_pre, "train_mssim_loss", train_MSSSIM_loss, global_rank)
    serialize_loss_item(dir_pre, "train_vgg_loss", train_vgg_loss, global_rank)

    serialize_loss_item(dir_pre, "val_mse_loss", val_MSE_loss, global_rank)
    serialize_loss_item(dir_pre, "val_total_loss", val_total_loss, global_rank)
    serialize_loss_item(dir_pre, "val_mssim_loss", val_MSSSIM_loss, global_rank)
    serialize_loss_item(dir_pre, "val_vgg_loss", val_vgg_loss, global_rank)

def _epoch(model, train_index_list, val_index_list, train_loader, val_loader, scheduler, optimizer, scaler, sched_type="expo"):
        train_MSE_loss = []
        train_MSSSIM_loss = []
        train_total_loss = []

        val_total_loss = []
        val_MSE_loss = []
        val_MSSSIM_loss = []
        print(f"initating training with a list of lenght len:{len(train_index_list)}")
        for index in range(0,len(train_index_list),  batch_size):
            # print(f"fetching first { batch_size} items from index: {index}: ")
            # print(f"fetching indices: {train_index_list[index: index+  batch_size]}")
            sample_batched =  train_loader.get_item(train_index_list[index: index+  batch_size])
            # print(f"item recieved:  {sample_batched}")
            HQ_img, LQ_img, maxs, mins, file_name = sample_batched['HQ'], sample_batched['LQ'], \
                sample_batched['max'], sample_batched['min'], sample_batched['vol']
            # print('indexes: ', idx)
            # print('shape: ', HQ_img.shape)
            # print('device: ', HQ_img.get_device())
            # print("got items")
            targets = HQ_img
            inputs = LQ_img
            with amp.autocast(enabled= amp_enabled):
                outputs =  model(inputs)
                MSE_loss = nn.MSELoss()(outputs, targets)
                MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
                loss = MSE_loss + 0.1 * (MSSSIM_loss)
                # print(loss)
            # print('calculating backpass')
            train_MSE_loss.append(MSE_loss.item())
            train_MSSSIM_loss.append(MSSSIM_loss.item())
            train_total_loss.append(loss.item())
            # model.zero_grad()
            optimizer.zero_grad(set_to_none=True)
            # BW pass
            if  amp_enabled:
                # print('bw pass')
                 scaler.scale(loss).backward()
                 scaler.step( optimizer)
                 scaler.update()
            else:
                #                 print('loss_bacl')
                loss.backward()
                #                 print('optimia')
                optimizer.step()
        print("schelud")
        if  sched_type == "platu":
            scheduler.step(loss)
        else:
            scheduler.step()

        print("Validation")
        for index in range(0,len(val_index_list),  batch_size):
            sample_batched =  val_loader.get_item(val_index_list[index: index+  batch_size])
            HQ_img, LQ_img, maxs, mins, fname = sample_batched['HQ'], sample_batched['LQ'], sample_batched['max'], sample_batched['min'], sample_batched['vol']

            with amp.autocast(enabled= amp_enabled):
                outputs =  model(LQ_img)
                MSE_loss = nn.MSELoss()(outputs, HQ_img)
                MSSSIM_loss = 1 - MSSSIM()(outputs, HQ_img)
                # loss = nn.MSELoss()(outputs , targets_val) + 0.1*(1-MSSSIM()(outputs,targets_val))
                loss = MSE_loss + 0.1 * (MSSSIM_loss)

            val_MSE_loss.append(MSE_loss.item())
            val_total_loss.append(loss.item())
            val_MSSSIM_loss.append(MSSSIM_loss.item())
        return train_total_loss, train_MSE_loss, train_MSSSIM_loss, val_total_loss, val_MSE_loss, val_MSSSIM_loss


def _epoch_vgg(model, train_index_list, val_index_list, train_loader, val_loader, scheduler, optimizer, scaler,
           sched_type="expo"):
    train_MSE_loss = []
    train_MSSSIM_loss = []
    train_total_loss = []
    train_vgg_loss = []

    val_total_loss = []
    val_MSE_loss = []
    val_MSSSIM_loss = []
    val_vgg_loss = []

    print(f"initating training with a list of lenght len:{len(train_index_list)}")
    for index in range(0, len(train_index_list), batch_size):
        # print(f"fetching first { batch_size} items from index: {index}: ")
        # print(f"fetching indices: {train_index_list[index: index+  batch_size]}")
        sample_batched = train_loader.get_item(train_index_list[index: index + batch_size])
        # print(f"item recieved:  {sample_batched}")
        HQ_img, LQ_img, maxs, mins, file_name = sample_batched['HQ'], sample_batched['LQ'], \
            sample_batched['max'], sample_batched['min'], sample_batched['vol']
        # print('indexes: ', idx)
        # print('shape: ', HQ_img.shape)
        # print('device: ', HQ_img.get_device())
        # print("got items")

        with amp.autocast(enabled=amp_enabled):
            outputs, out_b3, out_b1, tar_b3, tar_b1 = model(LQ_img, HQ_img)
            MSE_loss = nn.MSELoss()(outputs, HQ_img)
            MSSSIM_loss = 1 - MSSSIM()(outputs, HQ_img)
            loss_vgg_b1 = VGGloss()(out_b1,tar_b1)  # enhanced image : [1, 256, 56, 56] dim should be same (1,256,56,56)
            loss_vgg_b3 = VGGloss()(out_b3, tar_b3)
            loss_vgg = (loss_vgg_b3 + loss_vgg_b1)
            # loss = nn.MSELoss()(outputs , targets_val) + 0.1*(1-MSSSIM()(outputs,targets_val))
            loss = MSE_loss + gamma * (MSSSIM_loss) + beta * loss_vgg
            # print(loss)
        # print('calculating backpass')
        train_MSE_loss.append(MSE_loss.item())
        train_MSSSIM_loss.append(MSSSIM_loss.item())
        train_vgg_loss.append(loss_vgg.item())
        train_total_loss.append(loss.item())
        # model.zero_grad()
        optimizer.zero_grad(set_to_none=True)
        # BW pass
        if amp_enabled:
            # print('bw pass')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            #                 print('loss_bacl')
            loss.backward()
            #                 print('optimia')
            optimizer.step()
    print("schelud")
    if sched_type == "platu":
        scheduler.step(loss)
    else:
        scheduler.step()

    print("Validation")
    for index in range(0, len(val_index_list), batch_size):
        sample_batched = val_loader.get_item(val_index_list[index: index + batch_size])
        HQ_img, LQ_img, maxs, mins, fname = sample_batched['HQ'], sample_batched['LQ'], sample_batched['max'], \
        sample_batched['min'], sample_batched['vol']

        with amp.autocast(enabled=amp_enabled):
            outputs,out_b3, out_b1, tar_b3, tar_b1  = model(LQ_img, HQ_img)
            MSE_loss = nn.MSELoss()(outputs, HQ_img)
            MSSSIM_loss = 1 - MSSSIM()(outputs, HQ_img)
            # loss = nn.MSELoss()(outputs , targets_val) + 0.1*(1-MSSSIM()(outputs,targets_val))
            loss_vgg_b1 = torch.mean(torch.abs(torch.sub(out_b3,
                                                         tar_b3)))  # enhanced image : [1, 256, 56, 56] dim should be same (1,256,56,56)
            loss_vgg_b3 = torch.mean(torch.abs(torch.sub(out_b1,
                                                         tar_b1)))
            vgg_loss = (loss_vgg_b3 + loss_vgg_b1)
            # loss = nn.MSELoss()(outputs , targets_val) + 0.1*(1-MSSSIM()(outputs,targets_val))
            loss = MSE_loss + gamma * (MSSSIM_loss) + beta * vgg_loss

        val_MSE_loss.append(MSE_loss.item())
        val_total_loss.append(loss.item())
        val_vgg_loss.append(vgg_loss.item())
        val_MSSSIM_loss.append(MSSSIM_loss.item())
    return train_total_loss, train_MSE_loss, train_MSSSIM_loss, train_vgg_loss, val_total_loss, val_MSE_loss, val_MSSSIM_loss, val_vgg_loss


def _epoch_profile(model, train_index_list, val_index_list, train_loader, val_loader, scheduler, optimizer, scaler, sched_type):
        train_MSE_loss = [0]
        train_MSSSIM_loss = [0]
        train_total_loss = [0]
        val_total_loss = [0]
        val_MSE_loss = [0]
        val_MSSSIM_loss = [0]
        for index in range(0,len(train_index_list),  batch_size):
            sample_batched =  train_loader.get_item(train_index_list[index: index+  batch_size])
            HQ_img, LQ_img, maxs, mins, file_name = sample_batched['HQ'], sample_batched['LQ'], \
                sample_batched['max'], sample_batched['min'], sample_batched['vol']
            optimizer.zero_grad(set_to_none=True)
            targets = HQ_img
            inputs = LQ_img
            torch.cuda.nvtx.range_push("Training loop:  ")
            torch.cuda.nvtx.range_push("Forward pass")
            with amp.autocast(enabled= amp_enabled):
                outputs =  model(inputs)
                torch.cuda.nvtx.range_push("Loss calculation")
                MSE_loss = nn.MSELoss()(outputs, targets)
                MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
                loss = MSE_loss + 0.1 * (MSSSIM_loss)
                torch.cuda.nvtx.range_pop()
                print(loss)
            torch.cuda.nvtx.range_pop()

            train_MSE_loss.append(MSE_loss.item())
            train_MSSSIM_loss.append(MSSSIM_loss.item())
            train_total_loss.append(loss.item())
            torch.cuda.nvtx.range_push("backward pass")
            # BW pass
            if  amp_enabled:
                # print('bw pass')
                 scaler.scale(loss).backward()
                 scaler.step( optimizer)
                 scaler.update()
            else:
                loss.backward()
                optimizer.step()
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
        print("schelud")
        scheduler.step()
        print("Validation")
        torch.cuda.nvtx.range_push("Validation ")
        for index in range(0,len(val_index_list),  batch_size):
            sample_batched =  val_loader.get_item(val_index_list[index: index+  batch_size])
            HQ_img, LQ_img, maxs, mins, fname = sample_batched['HQ'], sample_batched['LQ'], \
                sample_batched['max'], sample_batched['min'], sample_batched['vol']
            inputs = LQ_img
            targets = HQ_img
            with amp.autocast(enabled= amp_enabled):
                outputs =  model(inputs)
                # outputs = model(inputs)
                MSE_loss = nn.MSELoss()(outputs, targets)
                MSSSIM_loss = 1 - MSSSIM()(outputs, targets)
                # loss = nn.MSELoss()(outputs , targets_val) + 0.1*(1-MSSSIM()(outputs,targets_val))
                loss = MSE_loss + 0.1 * (MSSSIM_loss)

            val_MSE_loss.append(MSE_loss.item())
            val_total_loss.append(loss.item())
            val_MSSSIM_loss.append(MSSSIM_loss.item())
        torch.cuda.nvtx.range_pop()
        return train_total_loss, train_MSE_loss, train_MSSSIM_loss, val_total_loss, val_MSE_loss, val_MSSSIM_loss

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

