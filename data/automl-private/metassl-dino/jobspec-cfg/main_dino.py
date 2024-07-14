# Copyright (c) Facebook, Inc. and its affiliates.
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
import argparse
import os
import sys
import datetime
import time
import math
import json
import neps
import logging
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Subset
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

from configspaces import get_pipeline_space
from eval_linear import eval_linear

import utils
import vision_transformer as vits
from vision_transformer import DINOHead
from functools import partial

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument("--batches_per_optimization_step", default=1, type=int,
        help="Number of batches that are used for one optimization step.")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # NEPS
    parser.add_argument("--is_neps_run", action="store_true", help="Set this flag to run a NEPS experiment.")
    parser.add_argument("--config_space", default="data_augmentation",
                        choices=["data_augmentation", "training", "groupaugment", "joint"],
                        help="Select the configspace you want to optimize with NEPS")
    parser.add_argument("--is_multifidelity_run", action="store_true", help="Store true if you want to activate multifidelity for NEPS")
    parser.add_argument("--use_fixed_DA_hypers", action="store_true", help="Store true if you want to start runs with a specific data augmentation configuration found by NEPS. Default hyperparameters will be overwritten for that run.")
    parser.add_argument('--valid_size', type=float, default=0.1, help="Define how much data to pick from the train data as val data. 0.1 means 10%")
    parser.add_argument('--dataset_percentage_usage', type=float, default=100, help="Define how much of your data to use. 100 means 100%. Will also influence the val data.")
    parser.add_argument('--train_dataset_percentage_usage', type=float, default=1, help="Define how much of your train data to use. 1 means 100%. Will not influence the val data.")
    parser.add_argument("--use_val_as_val", action="store_true", help="Use the validation set from ImageNet (which corresponds to the test set) as validation set. Use Test V2 as test set.")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--world_size", default=8, type=int, help="default is for NEPS mode with DDP, so 8.")
    parser.add_argument("--gpu", default=8, type=int, help="default is for NEPS mode with DDP, so 8 GPUs.")
    parser.add_argument('--config_file_path', help="Should be set to a path that does not exist.")
    parser.add_argument('--dataset', default='ImageNet', choices=['ImageNet', 'CIFAR-10', 'CIFAR-100', 'DermaMNIST', 'cars', 'flowers', 'inaturalist18', 'inaturalist19'],
                        help='Select the dataset on which you want to run the pre-training. Default is ImageNet')
    parser.add_argument("--use_imagenet_subset", action="store_true", help="Use a subset of the ImageNet dataset (containing 10% of the classes).")
    return parser


def dino_neps_main(working_directory, previous_working_directory, args, **hyperparameters):
    args.output_dir = working_directory
    ngpus_per_node = torch.cuda.device_count()
    print(f"Number of GPUs per node detected: {ngpus_per_node}")
    
    if args.is_neps_run:
        train_dino(torch.distributed.get_rank(), working_directory, previous_working_directory, args, hyperparameters)

        if torch.distributed.get_rank() == 0: # assumption: rank, running neps is 0
            # Return validation metric
            with open(str(args.output_dir) + "/current_val_metric.txt", "r") as f:
                val_metric = f.read()
            print(f"val_metric: {val_metric}")
            return -float(val_metric)  # Remember: NEPS minimizes the loss!!!
        return 0
    else:
        os.environ["WORLD_SIZE"] = str(args.world_size)
        print("test")
        train_dino(None, args.output_dir, args.output_dir, args)
        

def train_dino(rank, working_directory, previous_working_directory, args, hyperparameters=None):
    if not args.is_neps_run:
        print(f"init distributed mode executed")
        utils.init_distributed_mode(args, rank)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    
    cudnn.benchmark = True
    
    # ============ DINO run with NEPS ============
    if args.is_neps_run:
        args_dict = dict(vars(args))
        for k, v in hyperparameters.items():
            if k in args_dict:
                print(f"{k} : {args_dict[k]} (default: {v}) \n")
            else:
                print(f"{k} : {v}) \n")
                
        print("NEPS training hyperparameters: ", hyperparameters)
        
        # Parameterize hyperparameters
        if args.config_space == "training" or args.config_space == "joint":
            args.lr = hyperparameters["lr"]
            args.out_dim = hyperparameters["out_dim"]
            args.momentum_teacher = hyperparameters["momentum_teacher"]
            args.warmup_teacher_temp = hyperparameters["warmup_teacher_temp"]
            args.warmup_teacher_temp_epochs = hyperparameters["warmup_teacher_temp_epochs"]
            args.weight_decay = hyperparameters["weight_decay"]
            args.weight_decay_end = hyperparameters["weight_decay_end"]
            args.freeze_last_layer = hyperparameters["freeze_last_layer"]
            args.warmup_epochs = hyperparameters["warmup_epochs"]
            args.min_lr = hyperparameters["min_lr"]
            args.drop_path_rate = hyperparameters["drop_path_rate"]
            args.optimizer = hyperparameters["optimizer"]
            # args.use_bn_in_head = hyperparameters["use_bn_in_head"]
            # args.norm_last_layer = hyperparameters["norm_last_layer"]
        
        # if args.config_space == "data_augmentation" or args.config_space == "joint":
        #     args.local_crops_number = hyperparameters["local_crops_number"]

    else:
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # Fixes bug araising when testing for few epochs
    if args.warmup_epochs > args.epochs:
        args.warmup_epochs = args.epochs

    # ============ preparing data ... ============
    if args.config_space == "groupaugment":
        from data_augmentation import GroupAugmentDataAugmentationDINO
        transform = GroupAugmentDataAugmentationDINO(
            args.dataset,
            args.global_crops_scale,
            args.local_crops_scale,
            args.local_crops_number,
            args.is_neps_run,
            args.use_fixed_DA_hypers,
            hyperparameters,
            args.config_space,
        )
    else:
        transform = DataAugmentationDINO(
            args.dataset,
            args.global_crops_scale,
            args.local_crops_scale,
            args.local_crops_number,
            args.is_neps_run,
            args.use_fixed_DA_hypers,
            hyperparameters,
            args.config_space,
        )
    
    if args.dataset == "ImageNet" and args.use_imagenet_subset:
            args.data_path = "/work/dlclarge2/wagnerd-metassl-experiments/datasets/ImageNetSubset/10percent/train"
    dataset = utils.get_dataset(args=args, transform=transform, mode="train", pretrain=True)
    valid_size = args.valid_size if args.is_neps_run else 0  # default: 0.1 for 10%
    dataset_percentage_usage = args.dataset_percentage_usage  # default: 100 for 100%  # ToDo: Clean
    train_dataset_percentage_usage = args.train_dataset_percentage_usage  # default: 1 for 100%
    num_train = int(len(dataset) / 100 * dataset_percentage_usage)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    if args.is_neps_run and not args.use_val_as_val:
        if args.dataset == "ImageNet":
            if np.isclose(valid_size, 0.0):
                train_idx, valid_idx = indices, indices
            else:
                train_idx, valid_idx = utils.stratified_split(dataset.targets if hasattr(dataset, 'targets') else list(dataset.labels), valid_size, args.dataset)
                print("using balanced validation set")
                
                # use less train data
                if args.train_dataset_percentage_usage != 1:  # 1 means 100%
                    num_train_2 = int(len(train_idx))
                    indices_2 = list(range(num_train_2))
                    split_2 = int(np.floor(train_dataset_percentage_usage * num_train_2))
                    _, train_idx = indices[split_2:], indices[:split_2]
        else:
            # use val as val and test v2 as test
            if args.dataset == "ImageNet" and args.use_val_as_val:
                valid_size = 0
                print("Use val as val and test v2 as test")
                
            np.random.shuffle(indices)
            if np.isclose(valid_size, 0.0):
                train_idx, valid_idx = indices, indices
            else:
                train_idx, valid_idx = indices[split:], indices[:split]
            
            # use less train data
            if train_dataset_percentage_usage != 1:   # 1 means 100%
                num_train_2 = int(len(train_idx))
                indices_2 = list(range(num_train_2))
                split_2 = int(np.floor(train_dataset_percentage_usage * num_train_2))
                _, train_idx = indices[split_2:], indices[:split_2]

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_idx)
    
    else:
        # use less train data
        if train_dataset_percentage_usage != 1:   # 1 means 100%
            np.random.shuffle(indices)
            if np.isclose(valid_size, 0.0):
                train_idx, valid_idx = indices, indices
            else:
                train_idx, valid_idx = indices[split:], indices[:split]
            num_train_2 = int(len(train_idx))
            indices_2 = list(range(num_train_2))
            split_2 = int(np.floor(train_dataset_percentage_usage * num_train_2))
            _, train_idx = indices[split_2:], indices[:split_2]
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_idx)
        else:
            train_sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    if args.is_neps_run and not args.use_val_as_val:
        print(f"Data loaded: there are {len(train_idx)} images.")
    else:
        print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * args.batches_per_optimization_step * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if previous_working_directory is not None:
        utils.restart_from_checkpoint(
            os.path.join(previous_working_directory, "checkpoint.pth"), 
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
        )
    else:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, "checkpoint.pth"),  # for DINO baseline
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            dino_loss=dino_loss,
        )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    
    if args.is_neps_run and args.is_multifidelity_run:
        end_epoch = hyperparameters["epoch_fidelity"]
    else:
        end_epoch = args.epochs
   
    for epoch in range(start_epoch, end_epoch):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.is_neps_run:
        print("\n\n\nStarting Finetuning\n\n\n")
        finetuning_parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
        finetuning_parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
        finetuning_parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
        finetuning_parser.add_argument('--seed', default=0, type=int, help='Random seed.')
        finetuning_parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
        finetuning_parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
        finetuning_parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
        finetuning_parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
        finetuning_parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
        finetuning_parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
        finetuning_parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
        finetuning_parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
        finetuning_parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
        finetuning_parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
        finetuning_parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
        finetuning_parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
        finetuning_parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
        finetuning_parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
        finetuning_parser.add_argument("--is_neps_run", action="store_true", help="Set this flag to run a NEPS experiment.")
        finetuning_parser.add_argument("--config_space", default="data_augmentation", choices=["data_augmentation", "training", "groupaugment", "joint"], help="Select the configspace you want to optimize with NEPS")
        finetuning_parser.add_argument("--do_early_stopping", action="store_true", help="Set this flag to take the best test performance - Default by the DINO implementation.")
        finetuning_parser.add_argument("--world_size", default=8, type=int, help="actually not needed here -- just for avoiding unrecognized arguments error")
        finetuning_parser.add_argument("--gpu", default=8, type=int, help="actually not needed here -- just for avoiding unrecognized arguments error")
        finetuning_parser.add_argument('--config_file_path', help="actually not needed here -- just for avoiding unrecognized arguments error")
        finetuning_parser.add_argument('--warmup_epochs', help="actually not needed here -- just for avoiding unrecognized arguments error")
        finetuning_parser.add_argument('--dataset', default='ImageNet', choices=['ImageNet', 'CIFAR-10', 'CIFAR-100', 'DermaMNIST'], help='Select the dataset on which you want to run the pre-training. Default is ImageNet')
        finetuning_parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
        finetuning_parser.add_argument('--valid_size', type=float, default=0.1, help="Define how much data to pick from the train data as val data. 0.1 means 10%")
        finetuning_parser.add_argument('--dataset_percentage_usage', type=float, default=100, help="Define how much of your data to use. 100 means 100%. Will also influence the val data.")
        finetuning_parser.add_argument('--train_dataset_percentage_usage', type=float, default=1, help="Define how much of your train data to use. 1 means 100%. Will not influence the val data.")
        finetuning_parser.add_argument("--use_val_as_val", action="store_true", help="Use the validation set from ImageNet (which corresponds to the test set) as validation set. Use Test V2 as test set.")
        finetuning_parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
        finetuning_parser.add_argument("--use_imagenet_subset", action="store_true", help="Use a subset of the ImageNet dataset (containing 10% of the classes).")
        finetuning_args = finetuning_parser.parse_args()
        
        finetuning_args.arch = args.arch
        if args.dataset == "ImageNet" and args.use_imagenet_subset:
            finetuning_args.data_path = "/work/dlclarge2/wagnerd-metassl-experiments/datasets/ImageNetSubset/10percent/"
            finetuning_args.val_freq = 10
        else:
            finetuning_args.data_path = "/data/datasets/ImageNet/imagenet-pytorch/"
        finetuning_args.use_imagenet_subset = args.use_imagenet_subset
        finetuning_args.output_dir = args.output_dir
        finetuning_args.is_neps_run = args.is_neps_run
        finetuning_args.config_space = args.config_space
        finetuning_args.gpu = args.gpu
        finetuning_args.saveckp_freq = args.saveckp_freq
        finetuning_args.pretrained_weights = str(finetuning_args.output_dir) + "/checkpoint.pth"
        finetuning_args.seed = args.seed
        if not args.use_val_as_val:
            finetuning_args.assert_valid_idx = valid_idx
            finetuning_args.assert_train_idx = train_idx
        finetuning_args.use_val_as_val = args.use_val_as_val
        finetuning_args.valid_size = args.valid_size
        finetuning_args.dataset_percentage_usage = args.dataset_percentage_usage
        finetuning_args.train_dataset_percentage_usage = args.train_dataset_percentage_usage
        finetuning_args.local_crops_number = args.local_crops_number

        finetuning_args.dataset = args.dataset
        finetuning_args.epochs = 100
        if args.is_multifidelity_run:
            finetuning_args.epoch_fidelity = hyperparameters["epoch_fidelity"]
        
        eval_linear(finetuning_args)
            

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        do_optimizer_step = it % args.batches_per_optimization_step == args.batches_per_optimization_step - 1
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            raise ValueError("Loss value is invalid.")

        loss = loss / args.batches_per_optimization_step
        
        param_norms = None
        if fp16_scaler is None:
            
            loss.backward()
            
            if do_optimizer_step:
                if args.clip_grad:
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student,
                                                  args.freeze_last_layer)
                
                optimizer.step()
        else:
            
            fp16_scaler.scale(loss).backward()
            
            if do_optimizer_step:
                if args.clip_grad:
                    fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student,
                                                  args.freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
        
        if do_optimizer_step:
            # for name, p in student.named_parameters():
            #     if p.grad is not None:
            #         print(name, p.grad)
            #         break
            
            # student update
            optimizer.zero_grad()
            
            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item() * args.batches_per_optimization_step)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, dataset, global_crops_scale, local_crops_scale, local_crops_number, is_neps_run, use_fixed_DA_hypers, hyperparameters=None, config_space=None):
        if is_neps_run and (config_space == "data_augmentation" or config_space == "joint"):
            crops_scale_boundary = hyperparameters["crops_scale_boundary"]
            global_crops_scale = (crops_scale_boundary, global_crops_scale[1])
            local_crops_scale = (local_crops_scale[0], crops_scale_boundary)
            # local_crops_number = hyperparameters["local_crops_number"]

            p_horizontal_crop_1 = hyperparameters["p_horizontal_crop_1"]
            p_colorjitter_crop_1 = hyperparameters["p_colorjitter_crop_1"]
            p_grayscale_crop_1 = hyperparameters["p_grayscale_crop_1"]
            p_gaussianblur_crop_1 = hyperparameters["p_gaussianblur_crop_1"]
            p_solarize_crop_1 = hyperparameters["p_solarize_crop_1"]
            
            p_horizontal_crop_2 = hyperparameters["p_horizontal_crop_2"]
            p_colorjitter_crop_2 = hyperparameters["p_colorjitter_crop_2"]
            p_grayscale_crop_2 = hyperparameters["p_grayscale_crop_2"]
            p_gaussianblur_crop_2 = hyperparameters["p_gaussianblur_crop_2"]
            p_solarize_crop_2 = hyperparameters["p_solarize_crop_2"]
    
            p_horizontal_crop_3 = hyperparameters["p_horizontal_crop_3"]
            p_colorjitter_crop_3 = hyperparameters["p_colorjitter_crop_3"]
            p_grayscale_crop_3 = hyperparameters["p_grayscale_crop_3"]
            p_gaussianblur_crop_3 = hyperparameters["p_gaussianblur_crop_3"]
            p_solarize_crop_3 = hyperparameters["p_solarize_crop_3"]

            print("\n\nNEPS DATA AUGMENTATION HYPERPARAMETERS:\n")
            print(f"global_crops_scale: {global_crops_scale}")
            print(f"local_crops_scale: {local_crops_scale}")
            # print(f"local_crops_number: {local_crops_number}")
            print(f"p_horizontal_crop_1: {p_horizontal_crop_1}")
            print(f"p_colorjitter_crop_1: {p_colorjitter_crop_1}")
            print(f"p_grayscale_crop_1: {p_grayscale_crop_1}")
            print(f"p_gaussianblur_crop_1: {p_gaussianblur_crop_1}")
            print(f"p_solarize_crop_1: {p_solarize_crop_1}")
            print(f"p_horizontal_crop_2: {p_horizontal_crop_2}")
            print(f"p_colorjitter_crop_2: {p_colorjitter_crop_2}")
            print(f"p_grayscale_crop_2: {p_grayscale_crop_2}")
            print(f"p_gaussianblur_crop_2: {p_gaussianblur_crop_2}")
            print(f"p_solarize_crop_2: {p_solarize_crop_2}")
            print(f"p_horizontal_crop_3: {p_horizontal_crop_3}")
            print(f"p_colorjitter_crop_3: {p_colorjitter_crop_3}")
            print(f"p_grayscale_crop_3: {p_grayscale_crop_3}")
            print(f"p_gaussianblur_crop_3: {p_gaussianblur_crop_3}")
            print(f"p_solarize_crop_3: {p_solarize_crop_3}")
        else:
            if use_fixed_DA_hypers:
                if dataset == "ImageNet":
                    # config id 23
                    crops_scale_boundary = 0.40
                    global_crops_scale = (crops_scale_boundary, global_crops_scale[1])
                    local_crops_scale = (local_crops_scale[0], crops_scale_boundary)
                    local_crops_number = 8

                    p_horizontal_crop_1, p_colorjitter_crop_1, p_grayscale_crop_1, p_gaussianblur_crop_1, p_solarize_crop_1 = 0.27, 0.92, 0.32, 0.88, 0.14
                    p_horizontal_crop_2, p_colorjitter_crop_2, p_grayscale_crop_2, p_gaussianblur_crop_2, p_solarize_crop_2 = 0.65, 0.76, 0.29, 0.07, 0.19
                    p_horizontal_crop_3, p_colorjitter_crop_3, p_grayscale_crop_3, p_gaussianblur_crop_3, p_solarize_crop_3 = 0.30, 0.67, 0.27, 0.37, 0.17
                    
                    # config id 17 (backup)
                    crops_scale_boundary = 0.27
                    global_crops_scale = (crops_scale_boundary, global_crops_scale[1])
                    local_crops_scale = (local_crops_scale[0], crops_scale_boundary)
                    local_crops_number = 8

                    p_horizontal_crop_1, p_colorjitter_crop_1, p_grayscale_crop_1, p_gaussianblur_crop_1, p_solarize_crop_1 = 0.41, 0.77, 0.26, 0.80, 0.18
                    p_horizontal_crop_2, p_colorjitter_crop_2, p_grayscale_crop_2, p_gaussianblur_crop_2, p_solarize_crop_2 = 0.73, 0.86, 0.43, 0.12, 0.25
                    p_horizontal_crop_3, p_colorjitter_crop_3, p_grayscale_crop_3, p_gaussianblur_crop_3, p_solarize_crop_3 = 0.64, 0.63, 0.10, 0.33, 0.07
                elif dataset == "CIFAR-10":
                    crops_scale_boundary = 0.35
                    global_crops_scale = (crops_scale_boundary, global_crops_scale[1])
                    local_crops_scale = (local_crops_scale[0], crops_scale_boundary)
                    local_crops_number = 5

                    p_horizontal_crop_1, p_colorjitter_crop_1, p_grayscale_crop_1, p_gaussianblur_crop_1, p_solarize_crop_1 = 0.76, 0.89, 0.07, 0.90, 0.33
                    p_horizontal_crop_2, p_colorjitter_crop_2, p_grayscale_crop_2, p_gaussianblur_crop_2, p_solarize_crop_2 = 0.01, 0.91, 0.59, 0.11, 0.17
                    p_horizontal_crop_3, p_colorjitter_crop_3, p_grayscale_crop_3, p_gaussianblur_crop_3, p_solarize_crop_3 = 0.75, 0.63, 0.00, 0.17, 0.27
                elif dataset == "CIFAR-100":
                    crops_scale_boundary = 0.38
                    global_crops_scale = (crops_scale_boundary, global_crops_scale[1])
                    local_crops_scale = (local_crops_scale[0], crops_scale_boundary)
                    local_crops_number = 8

                    p_horizontal_crop_1, p_colorjitter_crop_1, p_grayscale_crop_1, p_gaussianblur_crop_1, p_solarize_crop_1 = 0.43, 0.78, 0.05, 0.90, 0.11
                    p_horizontal_crop_2, p_colorjitter_crop_2, p_grayscale_crop_2, p_gaussianblur_crop_2, p_solarize_crop_2 = 0.35, 0.65, 0.31, 0.09, 0.17
                    p_horizontal_crop_3, p_colorjitter_crop_3, p_grayscale_crop_3, p_gaussianblur_crop_3, p_solarize_crop_3 = 0.44, 0.62, 0.45, 0.19, 0.04
                else:
                    raise NotImplementedError
            else:
                p_horizontal_crop_1, p_colorjitter_crop_1, p_grayscale_crop_1, p_gaussianblur_crop_1, p_solarize_crop_1 = 0.5, 0.8, 0.2, 1.0, 0.0
                p_horizontal_crop_2, p_colorjitter_crop_2, p_grayscale_crop_2, p_gaussianblur_crop_2, p_solarize_crop_2 = 0.5, 0.8, 0.2, 0.1, 0.2
                p_horizontal_crop_3, p_colorjitter_crop_3, p_grayscale_crop_3, p_gaussianblur_crop_3, p_solarize_crop_3 = 0.5, 0.8, 0.2, 0.5, 0.0
        
        print("\n\nNEPS DATA AUGMENTATION HYPERPARAMETERS:\n")
        print(f"global_crops_scale: {global_crops_scale}")
        print(f"local_crops_scale: {local_crops_scale}")
        # print(f"local_crops_number: {local_crops_number}")
        print(f"p_horizontal_crop_1: {p_horizontal_crop_1}")
        print(f"p_colorjitter_crop_1: {p_colorjitter_crop_1}")
        print(f"p_grayscale_crop_1: {p_grayscale_crop_1}")
        print(f"p_gaussianblur_crop_1: {p_gaussianblur_crop_1}")
        print(f"p_solarize_crop_1: {p_solarize_crop_1}")
        print(f"p_horizontal_crop_2: {p_horizontal_crop_2}")
        print(f"p_colorjitter_crop_2: {p_colorjitter_crop_2}")
        print(f"p_grayscale_crop_2: {p_grayscale_crop_2}")
        print(f"p_gaussianblur_crop_2: {p_gaussianblur_crop_2}")
        print(f"p_solarize_crop_2: {p_solarize_crop_2}")
        print(f"p_horizontal_crop_3: {p_horizontal_crop_3}")
        print(f"p_colorjitter_crop_3: {p_colorjitter_crop_3}")
        print(f"p_grayscale_crop_3: {p_grayscale_crop_3}")
        print(f"p_gaussianblur_crop_3: {p_gaussianblur_crop_3}")
        print(f"p_solarize_crop_3: {p_solarize_crop_3}")
        if dataset == "ImageNet":
            global_crop_size = 224
            local_crop_size = 96
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif dataset == "CIFAR-10":
            global_crop_size = 32
            local_crop_size = 16
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        elif dataset == "CIFAR-100":
            global_crop_size = 32
            local_crop_size = 16
            normalize = transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762))
        elif dataset == "flowers":
            global_crop_size = 224
            local_crop_size = 96
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif dataset == "cars":
            global_crop_size = 224
            local_crop_size = 96
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            raise NotImplementedError(f"Dataset '{args.dataset}' not implemented yet!")

        normalize = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            
            transforms.RandomHorizontalFlip(p=p_horizontal_crop_1),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=p_colorjitter_crop_1
                ),
            transforms.RandomGrayscale(p=p_grayscale_crop_1),
            
            utils.GaussianBlur(p=p_gaussianblur_crop_1),  # default: 1.0
            utils.Solarization(p=p_solarize_crop_1),  # default: 0.0
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(global_crop_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
    
            transforms.RandomHorizontalFlip(p=p_horizontal_crop_2),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=p_colorjitter_crop_2
                ),
            transforms.RandomGrayscale(p=p_grayscale_crop_2),
            
            utils.GaussianBlur(p=p_gaussianblur_crop_2),  # default: 0.1
            utils.Solarization(p=p_solarize_crop_2),  # default: 0.2
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_crop_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
    
            transforms.RandomHorizontalFlip(p=p_horizontal_crop_3),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=p_colorjitter_crop_3
                ),
            transforms.RandomGrayscale(p=p_grayscale_crop_3),
            
            utils.GaussianBlur(p=p_gaussianblur_crop_3),  # default: 0.5
            utils.Solarization(p=p_solarize_crop_3),  # default: 0.0
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    # ignore depreciations
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    os.environ["NCCL_DEBUG"] = "INFO"
    
    # DINO run with NEPS
    if args.is_neps_run:
        utils.init_distributed_mode(args, None)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        )
        pipeline_space = get_pipeline_space(args)
       
        #dino_neps_main = partial(dino_neps_main, args=args)
        def main():
            with Path(args.config_file_path).open('rb') as f:
                dct_to_load = pickle.load(f)
            hypers = dct_to_load['hypers']
            working_directory = dct_to_load['working_directory']
            previous_working_directory = dct_to_load['working_directory']
                
            return dino_neps_main(working_directory=working_directory, previous_working_directory=previous_working_directory,
                                  args=args, **hypers)


        def main_master(working_directory, previous_working_directory, **hypers):
            dct_to_dump = {"working_directory": working_directory, "previous_working_directory": previous_working_directory, "hypers": hypers}
            with Path(args.config_file_path).open('wb') as f:
                pickle.dump(dct_to_dump, f)
            torch.distributed.barrier()
            return main()
            
        
        def main_worker():
            torch.distributed.barrier()
            main()
            
        
        if torch.distributed.get_rank() == 0:
            if args.is_multifidelity_run:
                # Add "eta=4," and "early_stopping_rate=1,"
                raise NotImplementedError
            neps.run(
                run_pipeline=main_master,
                pipeline_space=pipeline_space,
                root_directory=args.output_dir,
                max_evaluations_total=10000,
                max_evaluations_per_run=1,
                overwrite_working_directory=False,
                ignore_errors=True,
                acquisition_sampler="evolution",
                # eta=4,
                # early_stopping_rate=1,
            )
        else:
            main_worker()

    # Default DINO run
    else:
        dino_neps_main(args.output_dir, previous_working_directory=None, args=args)
