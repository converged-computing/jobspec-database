from dataset import PretrainDataModule, DownstreamDataModule
from model import Pretrain, ModifiedPretrain, Downstream
from utils import EarlyStoppingWithWarmup

import torch
from torch import nn, optim, utils
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
import yaml

import argparse
import os
import sys


def run(args: argparse.Namespace):
    with open(args.experiment, 'r') as f:
        settings = yaml.safe_load(f)

    # Global Configuration
    experiment_name = settings.get('name')
    seed = settings.get('seed', 0)
    debug = settings.get('debug', False)
    mode = settings.get('mode')
    batch_size = settings.get('batch_size', 32)
    num_workers = settings.get('num_workers', 4)
    checkpoint_path = settings.get('checkpoint_path', 'checkpoints/')
    root_dir = settings.get('root_dir', 'data/')

    # Model Configuration
    model_kwargs = settings.get('model_kwargs', None)

    # Criterion Configuration
    criterion_kwargs = settings.get('criterion_kwargs', None)

    # Optimizer Configuration
    optimizer_kwargs = settings.get('optimizer_kwargs', None)

    # Train Configuration
    train_config = settings.get('train', dict())
    train_max_epochs = train_config.get('max_epochs', 1)
    train_frac = train_config.get('frac', 1.0)
    train_text_req = train_config.get('text_req', 'both')
    train_modified_model = train_config.get('modified_model', False)

    # Downtream Configuration
    downstream_config = settings.get('downstream', dict())
    downstream_max_epochs = downstream_config.get('max_epochs', 1)
    dataset = downstream_config.get('dataset')
    finetune = downstream_config.get('finetune', False)

    # Setup Reproducibility & Debugging
    pl.seed_everything(seed, workers=True)
    limit_train_batches = 10 if debug else 1.0
    limit_val_batches = 10 if debug else 1.0
    limit_test_batches = 10 if debug else 1.0

    # Setup W&B
    #wandb_dir = '__pycache__/'
    wandb_logger = WandbLogger(
        project=experiment_name,
        #save_dir=wandb_dir,
        mode='offline',
        settings=wandb.Settings(start_method="fork"))

    # Initialize Lightning Module and Data Module
    os.environ['TOKENIZERS_PARALLELISM'] = 'false' # https://github.com/huggingface/transformers/issues/5486
    #os.environ['NCCL_DEBUG'] = 'INFO' # https://github.com/Lightning-AI/lightning/issues/10471
    
    if mode == 'pretrain':
        module = ModifiedPretrain if train_modified_model else Pretrain
        model = module(
            model_kwargs=model_kwargs,
            criterion_kwargs=criterion_kwargs,
            optimizer_kwargs=optimizer_kwargs)
        mimic_cxr_root_dir = '/vast/vs2393/mlh_dataset/' # root_dir of MIMIC_CXR data is in /vast (Comment out to inherit otherwise)
        datamodule = PretrainDataModule(
            root_dir=mimic_cxr_root_dir,
            batch_size=batch_size, 
            num_workers=num_workers,
            frac=train_frac,
            separate_sections=train_modified_model)

    elif mode == 'downstream':
        model_checkpoint = os.path.join(checkpoint_path, experiment_name, 'pretrain.ckpt')
        model = Downstream(
            model_checkpoint=model_checkpoint,
            optimizer_kwargs=optimizer_kwargs,
            modified_model=train_modified_model,
            finetune=finetune)
        datamodule = DownstreamDataModule(
            dataset=dataset,
            root_dir=root_dir,
            batch_size=batch_size,
            num_workers=num_workers)
    elif mode == 'evaluate':
        raise NotImplementedError(mode)
    else:
        raise ValueError(f'Unknown mode {mode}')

    # Initialize Callbacks
    # Remove?
    early_stop_callback = EarlyStoppingWithWarmup(warmup=5, monitor='val_f1', patience=5, verbose=False, mode='max')
    dirpath = os.path.join(checkpoint_path, experiment_name)
    model_checkpoint = ModelCheckpoint(
        monitor='train_loss' if mode == 'pretrain' else 'val_loss',
        save_top_k=1, 
        mode='min',
        dirpath=dirpath,
        filename=mode)
    lr_monitor = LearningRateMonitor()

    if mode == 'pretrain':
        callbacks = [model_checkpoint, lr_monitor]
    else:
        callbacks = [model_checkpoint, lr_monitor, early_stop_callback]
    # Train Model
    trainer = pl.Trainer(
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        max_epochs=train_max_epochs if mode == 'pretrain' else downstream_max_epochs,
        accelerator='gpu',
        strategy='ddp',
        devices=-1,
        num_nodes=1,
        auto_select_gpus=True,
        callbacks=[model_checkpoint, lr_monitor],
        deterministic=True,
        logger=wandb_logger)
    
    print('Training model...')
    trainer.fit(model, datamodule=datamodule)
    print('Training finished.')
    # Evaluate Model
    if mode == 'downstream':
        print('Evaluating model...')
        trainer.test(model, datamodule)
        print('Evaluation finished.')
    sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run an experiment.')
    parser.add_argument(
        '--experiment',
        required=True,
        type=str,
        help='path to experiment file (YAML)',
        metavar='EXPR',
        dest='experiment')

    args = parser.parse_args()
    run(args)
