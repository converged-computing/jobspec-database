# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import os
import time
import json
import random
import argparse
import datetime
import numpy as np
import wandb

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor
from torch.distributed.elastic.multiprocessing.errors import record
from torch.profiler import profile, record_function, ProfilerActivity


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    parser.add_argument('--partial_gpu', action='store_true', help="Use partial gpus of one physical node")
    parser.add_argument('--no_aug', action='store_true', help='do not use standard augmentation')

    args, unparsed = parser.parse_known_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    config = get_config(args)

    return args, config

@record
def main(config):
    # wandb.init(config=config, group="DDP")
    if config.AUG.NO_AUG:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        from torchvision.transforms import InterpolationMode
        transform = transforms.Compose([
            transforms.Resize(size = 256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.DATA.MEAN,
                std=config.DATA.STD
            )
        ])
        dataset_train = datasets.ImageNet(config.DATA.DATA_PATH, split='train', transform=transform)
        dataset_val = datasets.ImageNet(config.DATA.DATA_PATH, split='val', transform=transform)
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=dist.get_world_size(), rank= dist.get_rank(), shuffle=True
        )
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )
        data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS, pin_memory=True)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val, batch_size=config.DATA.BATCH_SIZE, num_workers=config.DATA.NUM_WORKERS , pin_memory=True)
        mixup_fn = None
    else:     
        dataset_train, dataset_val, data_loader_train, data_loader_val, data_loader_val_adv, mixup_fn = build_loader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.LOCAL_RANK == 0:
        os.environ["WANDB_MODE"] = "dryrun"
        if config.TRAIN.AUTO_RESUME and resume_file:
            run = wandb.init(project="test-project", group = "William", name = config.MODEL.NAME, resume = True)
        else: 
            run = wandb.init(config = config, project="test-project", entity="swin-transformer-poly", group = "William", name = config.MODEL.NAME)
        print(run.settings._offline)
    else:
        run = None

    if run is not None:
        run.watch(model, log_freq=100)

    if config.MODEL.RESUME:
        print(config.MODEL.RESUME)
        ## TODO: add consistency evaluation 
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        # logger.info(f"Shift Size {config.DATA.SHIFT_SIZE}")
        # acc1_adv, acc5_adv, loss_adv = validate(config, data_loader_val_adv, model)
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test shifted images: {acc1_adv:.1f}%")
        if run is not None:
            run.log({"Initial Accuracy": acc1, "Initial Loss": loss, "Initial Top5 Accuracy": acc5})
            # run.log({"Initial Accuracy Shifted": acc1_adv, "Initial Loss Shifted": loss_adv, "Initial Top5 Accuracy Shifted": acc5_adv})
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        # load_pretrained(config, model_without_ddp, logger) #FIXME control some other way
        acc1, acc5, loss = validate(config, data_loader_val, model)
        # logger.info(f"Shift Size {config.DATA.SHIFT_SIZE}")
        # acc1_adv, acc5_adv, loss_adv = validate(config, data_loader_val_adv, model)
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test shifted images: {acc1_adv:.1f}%")
        if run is not None:
            # "InitialAccuracy": acc1, "InitialLoss": loss
            # wandb summary
            wandb.run.summary["InitialAccuracy"] = acc1
            wandb.run.summary["InitialLoss"] = loss
            wandb.run.summary["InitialTop5Accuracy"] = acc5
            # wandb.run.summary["InitialAccuracyShifted"] = acc1_adv
            # wandb.run.summary["InitialLossShifted"] = loss_adv
            # wandb.run.summary["InitialTop5AccuracyShifted"] = acc5_adv
        
        if run is not None:
            run.log({"test accuracy": acc1, "max accuracy": acc1, "test loss": loss, "test_epoch": -1})


            
    if config.EVAL_MODE:
        return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler, run)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)

        acc1, acc5, loss = validate(config, data_loader_val, model)
        # acc1_adv, acc5_adv, loss_adv = validate(config, data_loader_val_adv, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if run is not None:
            wandb.summary["MaxAccuracy"] = max_accuracy
            # run.log({"Accuracy": acc1, "Loss": loss, "Top5 Accuracy": acc5, "Max Accuracy": max_accuracy, "test_epoch": epoch})
            run.log({"test_accuracy": acc1, "test_loss": loss, "test_top5_accuracy": acc5, "test_epoch": epoch, 'test_max_accuracy': max_accuracy})
            # run.log({"test_adv_accuracy": acc1_adv, "test_adv_loss": loss_adv, "test_adv_top5_accuracy": acc5_adv, "test_adv_epoch": epoch})

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    if run is not None:
        run.finish()


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, run = None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
#     with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     schedule=torch.profiler.schedule(
#         wait=1,
#         warmup=1,
#         active=2),
#     on_trace_ready=trace_handler
# ) as p:
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
            # print("Curr Mem - ", torch.cuda.memory_allocated()/1e9)
            # print("Curr Mem - ", torch.cuda.max_memory_allocated()/1e9)
            

        # p.step()
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            # optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None   
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            if run is not None:
                run.log({"time": batch_time.val, "loss": loss_meter.val, "grad_norm": norm_meter.val, "loss_scale":scaler_meter.val, "mem":memory_used, "train_step": epoch*num_steps + idx, "train_epoch": epoch, "lr": lr, "wd": wd})

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("./trace_" + str(p.step_num) + ".json")


if __name__ == '__main__':
    args, config = parse_option()
    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    if args.partial_gpu:
        torch.cuda.set_device(0)
    else:
        torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    # seed = config.SEED  
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    # linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE/ 512.0
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE/512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE/ 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    # logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))


    main(config)
