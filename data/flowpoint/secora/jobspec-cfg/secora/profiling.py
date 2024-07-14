import os
import sys
from math import ceil
import argparse
import tempfile
import random
import datetime

import pdb

import numpy as np

import torch
import torch.nn.functional as F

import torch
from torch import profiler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import BiEmbeddingModel
from data import *
from config import load_config, overwrite_config
from losses import contrastive_loss, mrr
from tracking import *
from infer import *

from SM3 import SM3

'''
usage like:
    python secora/profiling.py configs/config.yml --debug --modes train --modes validation --batch_size 8
'''

def train_step(model, optim, batch, config, device='cpu'):
    # a step without syncing, simulates grad accum
    with model.no_sync():
        model_inputs = batch['input_ids'], batch['token_type_ids'], batch['attention_mask'],
        loss = contrastive_loss(model, model_inputs, config)
        loss.backward()

    # a step without syncing, simulates grad accums optimization step
    loss = contrastive_loss(model, model_inputs, config)
    loss.backward()

    optim.step()
    optim.zero_grad(set_to_none=True)


def profile(config, modes, **kwargs):
    rank = dist.get_rank()
    logger = kwargs['logger']

    #needed for warmup anyway
    train_set = preprocess_split(data.DataSplit.TRAIN, config, limit_samples=config['batch_size']*50, **kwargs)
    train_loader = get_loader(train_set, config)
    train_iter = iter(deviceloader(train_loader, rank))

    if 'validation' in modes or 'embedding' in modes:
        valid_set = preprocess_split(data.DataSplit.VALIDATION, config, limit_samples=config['batch_size']*5, **kwargs)
        valid_loader = get_loader(valid_set, config)

    logger.info('building model')
    m = BiEmbeddingModel(config).to(rank)
    #model = DDP(m, device_ids=[rank])

    # some warmup
    logger.info('warming up cuda benchmark on train set')
    for step, batch in zip(range(12), deviceloader(train_loader,rank)):
        model_inputs = batch['input_ids'], batch['token_type_ids'], batch['attention_mask']
        m(*model_inputs)

    if config['cuda_graphs'] == True:
        logger.info('cuda_graphs is True: building the cuda graph')
        #m = BiEmbeddingModel(config).to(rank)
        m.make_graphed(model_inputs)

    logger.info('building DDP model')
    model = DDP(m, device_ids=[rank])

    if config['finetune_mode'] == 'all':
        params = model.parameters()
    elif config['finetune_mode'] == 'pooling':
        params = model.pooling.parameters()
    else:
        raise RuntimeError('finetune_mode has to be: all or pooling')

    if config['optim'] == 'adam':
        optim = torch.optim.Adam(params, lr=config['lr'])
    elif config['optim'] == 'sgd':
        optim = torch.optim.SGD(params, lr=config['lr'])
    elif config['optim'] == 'sm3':
        optim = SM3(params, lr=config['lr'])
    else:
        raise RuntimeError('config specifies and unsupported optimizer')

    logger.info('starting profiler')

    tensorboard_run_path = os.path.join(config['logdir'], config['name'])
    trace_path = os.path.join(config['logdir'], config['name'], 'profile_trace.json')
    stacks_path = os.path.join(config['logdir'], config['name'], 'profile_stacks.txt')

    code_embedding = None
    doc_embedding = None

    torch.cuda.synchronize()
    dist.barrier()

    with_stack = False

    with profiler.profile(
        with_stack=with_stack,
        #profile_memory=True, 
        #record_shapes=True,
        on_trace_ready=profiler.tensorboard_trace_handler(tensorboard_run_path),
        schedule=torch.profiler.schedule(
            skip_first=0,
            wait=5,
            warmup=1,
            active=2),
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ]) as p:

        for idx in range(10):
            logger.info(f'step {idx}')
            if 'train' in modes:
                torch.cuda.synchronize()
                dist.barrier()
                model.train()
                logger.info(f'train_step')
                train_step(model, optim, next(train_iter), config, device=rank)
                torch.cuda.synchronize()
                dist.barrier()

            if 'embedding' in modes or ('validation' in modes and code_embedding is None):
                code_embedding = build_embedding_space(model, valid_loader, config, feature_prefix='code_', embedding_size=config['embedding_size'], device=rank, **kwargs)
                doc_embedding = build_embedding_space(model, valid_loader, config, feature_prefix='doc_', embedding_size=config['embedding_size'], device=rank, **kwargs)

            if 'validation' in modes:
                with model.no_sync():
                    with torch.no_grad():
                        model.eval()
                        logger.info(f'eval step')
                        distances, neighbors = k_nearest_neighbors(
                                doc_embedding,
                                code_embedding,
                                embedding_size=config['embedding_size'], 
                                top_k=config['top_k'],
                                logger=logger)
            p.step()

    torch.cuda.synchronize()
    dist.barrier()


    if rank == 0:
        print(p.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))
        if with_stack == True:
            p.export_stacks(stacks_path, "self_cuda_time_total")

    torch.cuda.synchronize()
    dist.barrier(group=dist.group.WORLD)
    dist.destroy_process_group()


def profiling_worker(rank, config, modes, debug, master_port):
    world_size = config['num_gpus']
    host_name = config['hostname']
    #port = config['port']

    os.environ['MASTER_ADDR'] = host_name
    os.environ['MASTER_PORT'] = master_port

    #os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    #os.environ.pop('MASTER_ADDR', None)
    #os.environ.pop('PORT', None)
    #os.environ.pop('NCCL_ASYNC_ERROR_HANDLING', None) 

    # this needs a filesystem that has fnctl
    #tmpdir = tempfile.TemporaryDirectory(prefix='/home/fhoels2s/tmp/') 
    # initialize the process group
    #dist.init_process_group('nccl', rank=rank, world_size=world_size, init_method='file://' + os.path.join(tmpdir.name, 'torch_sharedfile'), timeout=datetime.timedelta(65))
    dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=datetime.timedelta(65))
    logger = make_logger(config, debug=debug, rank=rank)
    torch.cuda.set_device(rank)
    profile(config, modes, progress=False, debug=debug, logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual profiling script.')
    parser.add_argument('config_path', type=str)
    parser.add_argument('--modes', type=str, action='append', default=[])
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()

    if args.modes == []:
        raise RuntimeError('--modes have to be at least one')

    config = load_config(args.config_path)
    config = overwrite_config(args, config)

    logdir = os.path.join(config['logdir'], config['name'])
    checkdir = os.path.join(config['checkpoint_dir'], config['name'])
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkdir, exist_ok=True)

    master_port = str(random.randint(10000, 15000))
    modes = args.modes

    avail_modes = ['train', 'embedding', 'validation']
    if any([m not in avail_modes for m in modes]):
        raise RuntimeError('only the modes train, embedding, validation are supported')

    np.random.seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.empty_cache()

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = True

    mp.set_start_method('spawn')
    mp.spawn(profiling_worker, 
            args=(config, modes, args.debug, master_port),
            nprocs = config['num_gpus'],
            join=True)
