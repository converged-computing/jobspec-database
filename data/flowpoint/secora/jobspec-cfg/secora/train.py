import sys
import os
from time import time, sleep
from enum import Enum, auto
from math import ceil
import re
from dataclasses import dataclass, field
from functools import partial

from datasets import load_from_disk

from pdb import set_trace as bp

import torch
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler

import torch.distributed as dist
import torch.multiprocessing as mp
import multiprocessing
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from secora.models import *
import secora.models as models

from secora.data import *
from secora import data
from secora.config import *
from secora.infer import validate
from secora.losses import contrastive_loss, mrr
from secora.tracking import * # init_storage
from secora.display import Display
from secora.metrics import MetricLogger
from secora.train_utils import *

from SM3 import SM3

'''
all things regarding a single "training run"
note that hyperparameter optimization (hparam search) creates multiple "training runs"
'''

class FinetuneMode(Enum):
    ALL = 'all'
    POOLING = 'pooling'

class ScheduleEnum(Enum):
    LINEAR = 'linear'
    CONSTANT = 'constant'
    
class OptimizerEnum(Enum):
    ADAM = 'adam'
    ADAMW = 'adamw'
    SGD = 'sgd'
    SM3 = 'sm3'

_optimizer_mapping = {'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'sm3': SM3
        }

class TrainingRunIDSetting(Setting):
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    @property
    def allowed_type(self):
        return str

    def check(self, val):
        return True

class TrainingConfig(SimpleConfig):
    def __init__(self):
        super().__init__()

        setts = [
            IntSetting('batch_size', lb=1),
            IntSetting('seed', lb=0),
            IntSetting('epochs', lb=1),
            IntSetting('shards', lb=1),
            IntSetting('grad_accum', lb=1),
            IntSetting('grad_cache', lb=1),
            IntSetting('warmup_batches',lb=0),
            FloatSetting('temp', lb=0., ub=1.),
            IntSetting('top_k', lb=0),
            DirectorySetting('checkpoint_dir'),
            IntSetting('max_checkpoints', lb=0),
            models.basemodel_setting(),
            FloatSetting('learning_rate', lb=0.),
            EnumSetting('finetune_mode', FinetuneMode),
            data.LanguageSetting('languages'),
            IntSetting('preprocess_cores', lb=1),
            EnumSetting('preprocess_mode', data.PreprocessMode),
            IntSetting('max_input_tokens', lb=1),
            EnumSetting('optimizer', OptimizerEnum),
            models.amp_setting(),
            EnumSetting('lr_schedule', ScheduleEnum),
            models.dropout_setting(),
            TrainingRunIDSetting('training_run_id'),
            IntSetting('num_gpus', lb=0),
            BoolSetting('cuda_graphs'),
            BoolSetting('deterministic'),
            IntSetting('embedding_size', lb=0),
            FloatSetting('grad_clip', lb=0.),
            DirectorySetting('logdir'),
            ]

        for s in setts:
            self.add(s)

    def parse_from_dict(self, config_candidate):
        for k,v in config_candidate.items():
            # parse through the settings parsing function
            self[k] = self.settings[k].parse(v)


def should_optimize(step, config):
    ''' wheter an optimizer step should be run '''
    #cache_accum = 512 // config['batch_size']
    #return (step+1) % cache_accum == 0
    cache_accum = 512 // config['batch_size']
    return (step+1) % cache_accum == 0
    #return True


def train_step(
        shard_iter,
        state_tracker,
        config, 
        **kwargs):
    ''' the necessary accumulation/gradcache steps 
    until and including one optimizer step 

    note: the last values in a shard are omitted
    because they won't fit for a full optimizer/accumulation step/batch

    this can be alleviated by fitting the shard size or optimizer/accumulation,
    so that the accumulation divides the shard size
    '''

    optim = state_tracker['optimizer']
    scheduler = state_tracker['scheduler']
    training_progress = state_tracker['training_progress']
    scaler = state_tracker['scaler']
    model = state_tracker['model']
    rank = kwargs['rank']

    model.train()

    cache = GCache()
    forward_1 = forward_2 = model

    def loss_fn(a, b):
        return contrastive_loss(a, b, temp=config['temp'])

    def optimize(*args, **kwargs):
        optim.step()
        optim.zero_grad(set_to_none=True)

    if config['grad_cache'] == True:
        fns = gradcache_ify(forward_1, forward_2, loss_fn, optimize, cache)
        forward_1, forward_2, loss_fn, optimize = fns

    step = 0
    step_loss = torch.tensor([0.], device=rank)

    while True:#not should_optimize(step, config) and step < len(train_loader):
        try:
            #batch = next(map(partial(batch_to_device, rank), train_loader))
            batch = next(shard_iter)
        except StopIteration as e:
            # cant complete the last shard optimizer step with the set batchsize
            return step_loss, True

        
        should_break = should_optimize(step, config)

        model_inputs = batch['input_ids'], batch['attention_mask']

        with torch.no_grad():
            emb1 = forward_1(*model_inputs)
        emb2 = forward_2(*model_inputs)
        loss = loss_fn(emb1, emb2)

        if config['amp'] != AMP.DISABLE:
            loss = scaler.scale(loss)

        step_loss += loss
        loss.backward()
        step += 1
        
        if should_break:
            break

    # only sync before optimizer step
    if kwargs.get('distributed', True) == True:
        torch.cuda.synchronize()
        dist.barrier()
    
    #gradient clipping
    if config['grad_clip'] != 0.:
        scaler.unscale_(optim)
        clip_grad_norm_(
                model.parameters(), 
                max_norm=config['grad_clip'])

    if config['amp'] != AMP.DISABLE:
        scaler.step(optim)
        scaler.update()
    else:
        optimize()

    scheduler.step()
    training_progress.step_done()# += 1

    return step_loss, False


def train_shard(
        state_tracker,
        train_loader,
        config,
        **kwargs
        ):

    rank = kwargs.get('rank', 0)
    logger = kwargs['logger']
    scheduler = state_tracker['scheduler']
    training_progress = state_tracker['training_progress']
    display = kwargs['display']
    metriclogger = kwargs['metriclogger']

    shard_loss = torch.tensor([0.], device=rank, dtype=torch.float64, requires_grad=False)
    shard_step = 0
    shard_done = False

    logger.debug(f'starting shard {training_progress.shard} of length: {len(train_loader)}')

    shard_iter = iter(map(partial(batch_to_device, rank), train_loader))

    display.start_shard() 
    global_step = training_progress.epoch*training_progress.shard*training_progress.step

    while shard_done == False:
        loss, shard_done = train_step(
                shard_iter,
                state_tracker,
                config, 
                **kwargs)

        logger.debug(f'step {shard_step}')

        #step_bar.update(config['grad_accum']*config['grad_cache']*config['batch_size'])
        display.update(training_progress)
        shard_step += 1

        if rank == 0:
            metriclogger.add_scalar(
                    "learning_rate/train", 
                    scheduler.get_last_lr()[0], 
                    global_step)
            metriclogger.flush()

        shard_loss.add_(loss.detach())
        shard_step += 1

    dist.all_reduce(shard_loss)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        avg_loss = shard_loss.cpu().numpy() / shard_step
        metriclogger.add_scalar("avg_loss/train", 
                avg_loss, 
                global_step)
        metriclogger.flush()


def train_epoch(model, train_set, valid_set, num_shards, training_progress, config, state_tracker, **kwargs):
    #config = kwargs['config']
    display = kwargs['display']
    rank = kwargs['rank']
    logger = kwargs['logger']
    logger.info(f'starting epoch: {training_progress.epoch}')
    train_set.shuffle()


    display.start_epoch()
    while(training_progress.shard < num_shards):
        logger.info(f'training shard: {training_progress.shard}')

        shard = train_set.shard(num_shards, training_progress.shard, contiguous=True)
        train_loader = get_loader(shard, config['batch_size'])

        train_shard(
            state_tracker,
            train_loader,
            config,
            **kwargs
            )

        logger.info(f'validating shard {training_progress.shard}')

        score = 0.
        score = validate(state_tracker['model'], 
                valid_set, 
                config, 
                state_tracker['training_progress'],
                **kwargs)

        training_progress.shard_done()

        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            state_tracker.save()

        #shard_bar.update()

        preempt_callback = kwargs.get('preempt_callback', None)
        if preempt_callback is not None:
            if not callable(preempt_callback):
                raise RuntimeError(f'preempt_callback has to be callable but was {preempt_callback}')
            preempt_callback(state_tracker, score, config, **kwargs)


    training_progress.epoch_done()

    return score


def hw_warmup(model, train_set, **kwargs):
    ''' 
    warmup hardware and accelerators
    collects data for lower level optimization algorithms
    done separately before training for reproducability with changing hw setups

    train_set needs to be without sideeffects when reading it
    '''

    logger = kwargs['logger']
    config = kwargs['config']
    rank = kwargs['rank']
    logger.info('warming up cuda benchmark')
    train_loader = get_loader(train_set, config['batch_size'], workers=0, dist=dist.is_initialized(), **kwargs)
    for step, batch in zip(range(12), map(partial(batch_to_device, rank), train_loader)):
        model_inputs = batch['input_ids'], batch['attention_mask']
        model(*model_inputs)
        torch.cuda.synchronize()
        dist.barrier()


def hw_optimize(model, **kwargs):
    '''
    use the warmup statistics to optimize performance before starting training
    '''

    config = kwargs['config']
    if config['cuda_graphs'] == True:
        logger.info('cuda_graphs is True: building the cuda graph')
        torch.cuda.synchronize()
        dist.barrier()
        model.make_graphed(model_inputs)
        torch.cuda.synchronize()
        dist.barrier()


def distribute_model(model, **kwargs):
    ''' 
    uses an automatic distribution algorithm to distribute a model architecture 
    if a certain distribution pattern is wanted, extend this function
    or write a custom distributed model
    '''

    kwargs['logger'].info('building distributed model')
    return DDP(model, device_ids=[kwargs['rank']])#, find_unused_parameters=True)

def build_optimizer(config, model, **kwargs):
    #mode: FinetuneMode = FinetuneMode(config['finetune_mode'])
    mode = config['finetune_mode']

    if mode is FinetuneMode.ALL:
        params = model.parameters()
    elif mode is FinetuneMode.POOLING:
        params = model.pooling.parameters()
    else:
        raise RuntimeError(f'invalid finetune_mode {mode}')

    optim = _optimizer_mapping[config['optimizer'].value](params, lr=config['learning_rate'])
    return optim

def build_scheduler(optim, config, train_set_len, **kwargs):
    num_warmup_steps = ceil(config['warmup_batches'] / config['grad_accum'])
    num_training_steps_per_epoch = ceil(train_set_len / (config['batch_size'] * config['grad_accum']))
    num_training_steps = config['epochs'] * num_training_steps_per_epoch

    if config['lr_schedule'] == ScheduleEnum.LINEAR:
        scheduler = get_linear_schedule_with_warmup(
                optim,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps)
    elif config['lr_schedule'] == ScheduleEnum.CONSTANT:
        scheduler = get_constant_schedule_with_warmup(
                optim,
                num_warmup_steps=num_warmup_steps
                )

    return scheduler


def train(
        config,
        resume,
        *args, 
        preempt_callback=None, 
        cache=False, 
        **kwargs):
    rank = kwargs['rank']
    torch.autograd.set_detect_anomaly(kwargs.get('debug', False))
    logger = kwargs['logger']
    logger.info('started train function')

    if kwargs['debug'] == True:
        t_limit = 20*config['grad_accum']*config['batch_size']
        v_limit = 20*config['grad_accum']*config['batch_size']
    else:
        t_limit = v_limit = 10000


    # training blocks

    logger.info('started preprocessing splits')
    # data
    train_path =  '/tmp/secora_ds/train_set'
    if cache == True:
        if os.path.exists(train_path):
            train_set = load_from_disk(train_path)
            train_set.set_format(type='torch', columns=list(set(train_set.column_names) - {'url','language'}), output_all_columns=True)
    else:
        train_set = preprocess_split(data.DataSplit.TRAIN, config, limit_samples=t_limit, **kwargs)
        train_set.save_to_disk(dataset_path=train_path)

    valid_path =  '/tmp/secora_ds/valid_set'
    if cache == True:
        if os.path.exists(valid_path):
            valid_set = load_from_disk(valid_path)
            valid_set.set_format(type='torch', columns=list(set(valid_set.column_names) - {'url','language'}), output_all_columns=True)
    else:
        valid_set = preprocess_split(data.DataSplit.VALIDATION, config, limit_samples=v_limit, **kwargs)
        valid_set.save_to_disk(dataset_path=valid_path)


    logger.info(f'finished preprocessing splits with train_len: {len(train_set)} valid_len: {len(valid_set)}')

    # initialize state
    m = build_model(config, **kwargs)

    # this should soon be rather done after the training initialization
    hw_warmup(m, train_set, config=config, **kwargs)
    hw_optimize(m, config=config, **kwargs)

    model = distribute_model(m, **kwargs)

    optim = build_optimizer(config, model)
    scheduler = build_scheduler(optim, config, len(train_set))
    training_progress = TrainingProgress()
    state_tracker = StateTracker(
            config['training_run_id'],
            config['logdir'],
            config['max_checkpoints'],
            logger,
            model=model,
            optimizer=optim,
            scheduler=scheduler,
            scaler=GradScaler(),
            training_progress=training_progress)

    if kwargs.get('debug', False) == True:
        num_epochs = 2
        num_shards = 2
    else:
        if resume == True:
            # load latest checkpoint 
            # resume by default
            torch.cuda.synchronize()
            dist.barrier()
            state_tracker.load_latest()


        num_epochs = config['epochs']
        num_shards = config['shards']

    display = kwargs['display']
    num_steps = len(train_set) / (num_epochs * num_shards * config['grad_accum'] * config['grad_cache'] * config['batch_size'])
    display.set_total(len(train_set), len(valid_set), num_epochs, num_shards, num_steps)

    logger.info(f'shard_size: {len(train_set)//num_shards} samples')
    logger.info(f'validation set size: {len(valid_set)} samples')
    logger.info(f'starting training')

    # do one validation pass with the base model
    '''
    score = validate(state_tracker['model'], 
            valid_set, 
            config, 
            state_tracker['training_progress'],
            **kwargs)
    '''

    display.start_training()
    while(training_progress.epoch < num_epochs):
        score = train_epoch(model, train_set, valid_set, num_shards, training_progress, config, state_tracker,
                preempt_callback=preempt_callback, **kwargs)

    display.close()

            #train_bar.update()

    if 'hparam_callback' in kwargs and rank == 0:
        kwargs['hparam_callback'](kwargs['metriclogger'], score, rank, config)

    return score
    #return torch.tensor([1.])#state_tracker


def training_worker(rank, return_values, config, resume, progress, args, kwargs):
    world_size = config['num_gpus']
    debug = kwargs.get('debug', False)

    #os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    init_storage(config, rank)
    logger = make_logger(config, debug=debug, rank=rank)

    display = Display(show_progress=progress, rank=rank)
    metriclogger = MetricLogger(config, rank)

    logger.info(f'starting run with training_run_id: {config["training_run_id"]}')

    kwargs['debug'] = debug
    logger.info('running debug mode')

    result = train(config, 
            resume,
            *args,
            display=display, 
            metriclogger=metriclogger, 
            progress=progress, 
            rank=rank, 
            logger=logger,
            **kwargs)

    logger.debug('finished train_fn')


    torch.cuda.synchronize()
    dist.barrier(group=dist.group.WORLD)

    #return_values.put({"rank": rank, "result": result})
    return_values.put({"rank": rank, "result": result})
        
    dist.destroy_process_group()


def build_config(config_candidate, *args, **kwargs):
    config = TrainingConfig()
    config.parse_from_dict(config_candidate)
    config.check()
    config.final()

    return config


def load_config(resume_args, *args, **kwargs):
    ''' 
    train_run_id can be arbitrary, but must be unique in the config/training run store 
    for simplicity, just set to timestamp for now
    '''

    training_run_id = resume_args.training_run_id

    config = TrainingConfig()

    storage_path = kwargs['storage_path']
    if storage_path is None:
        storage_path = '~/secora_output'

    # this will need to be parallelism safe in future
    with open(os.path.join(storage_path, training_run_id, 'config.yml'), 'r') as f:
        config_candidate = yaml.safe_load(f) 

    config.parse_from_dict(config_candidate)
    config.check()
    config.final()

    return config


def run_workers(config, resume, *args, **kwargs):
    # rng has to be inited before spawning
    # because multiprocessing requires different worker ids
    # that might be created randomly
    if kwargs['debug'] == True and config['deterministic'] == True:
        print('warning, debug nondeterministic run')

    rng_init(seed=config['seed'], deterministic=kwargs.get('deterministic', False))


    mp.set_start_method('spawn', force=True)
    return_values = mp.SimpleQueue()
    ctx = mp.spawn(training_worker, 
            args=(return_values, config, resume, kwargs.get('progress', True), args, kwargs),
            nprocs = config['num_gpus'],
            join=True)

    results = []
    for i in range(config['num_gpus']):
        results.append(return_values.get())

    return results


def train_start(config_args, *args, **kwargs):
    ''' 
    train_run_id can be arbitrary, but must be unique in the config/training run store 
    for simplicity, just set to timestamp for now
    '''
    training_run_id = str(ceil(time()))

    # this will need to be parallelism safe in future
    with config_args.config_file as f:
        config_candidate = yaml.safe_load(f)
        config_candidate['training_run_id'] = training_run_id

    if config_args.batch_size is not None:
        config_candidate['batch_size'] = config_args.batch_size

    if config_args.max_checkpoints is not None:
        config_candidate['max_checkpoints'] = config_args.max_checkpoints

    config_candidate['deterministic'] = config_args.deterministic

    config = build_config(config_candidate, *args, **kwargs)

    kwargs['debug'] = config_args.debug
    r = run_workers(config, resume=False, *args, **kwargs)

    return r


def train_resume(resume_args, *args, **kwargs):
    # run state_url is the path/url where the run state to be resumed resides
    config = load_config(
            resume_args, 
            *args, 
            storage_path=resume_args.storage_path, 
            **kwargs)
    run_workers(
            config, 
            resume=True, 
            *args, 
            debug=resume_args.debug, 
            **kwargs)
