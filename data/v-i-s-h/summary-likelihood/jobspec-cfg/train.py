# Train a Bayesian Neural Network with weight space prior using
# reparameterization


import os
import json
import argparse
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

import models
import datasets
import methods
import transforms
from utils import parse_params_str
import config as cfg


def run_experiment(
        method, method_params,
        dataset, ds_params, transform, 
        model_str, model_params,
        max_steps, batch_size,
        wt_loss, 
        mc_samples, 
        use_gpu,
        outdir):
    """
        Trains a model on dataset and returns model and logs

    Parameters
    ----------
    method : str
        Method to train BNN
    methods_params : str
        Parameters for method in comma separated list
    dataset : str
        Name of dataset
    ds_params : dict
        kwargs for dataset class
    transform : str
        Input transformation
    model_str : str
        Model name
    model_params: dict
        kwargs for model class
    max_steps : int
        Maximum number of epochs to train
    batch_size : int
        Minibatch size
    wt_loss : Bool
        Whether to use weighted loss for prediction loss
    mc_samples : int
        Number of Monte Carlo forwards for BNN
    use_gpu : Boolean
        Whether to use GPU or not

    """
    if use_gpu and torch.cuda.is_available():
        gpus = 1
    else:
        gpus = None

    disable_pbar = int(os.environ.get('DISABLE_PBAR', 0)) == 1

    x_transform = getattr(transforms, transform)() # returns a transformation object

    # Load dataset
    DatasetClass = getattr(datasets, dataset)
    trainset = DatasetClass(**ds_params, split='train', transform=x_transform)
    valset = DatasetClass(**ds_params, split='val', transform=x_transform)
    # Ignore size constraint for test set
    _ds_params = copy.deepcopy(ds_params)
    if 'size' in _ds_params:
        del _ds_params['size']
    testset = DatasetClass(**_ds_params, split='test', transform=x_transform)

    tr_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=valset, batch_size=8*batch_size, shuffle=False)
    test_loader = DataLoader(dataset=testset,  batch_size=8*batch_size, shuffle=False)

    N = len(trainset)
    K = trainset.n_labels

    # Build model
    ModelClass = getattr(models, model_str)

    # Check for weighted loss function
    if wt_loss:
        w = [1.0, trainset.pos_weight]
        print("INFO: Using weighted loss function.      w =", w)
    else:
        w = None

    # Prepare model to train
    model = ModelClass(K, **model_params)

    MethodClass = getattr(methods, method)

    # Get default params
    method_params = MethodClass.populate_missing_params(method_params, trainset)
    if isinstance(MethodClass, methods.BaseModel):
        pl_model = MethodClass(model, **method_params, class_weight=w, mc_samples=mc_samples)
    else:
        pl_model = MethodClass(model, **method_params, class_weight=w)

    tb_logger = pl_loggers.TensorBoardLogger(outdir, name="", version="tblog")
    tb_logger.log_hyperparams(pl_model.hparams)
    ckp_cb = ModelCheckpoint(outdir, 
                save_last=True, 
                save_top_k=1, monitor='val_f1', mode='max',
                filename="{step:05d}")

    trainer = Trainer(
        max_steps=max_steps,
        gpus=gpus,
        logger=tb_logger,
        callbacks=[ckp_cb],
        enable_progress_bar=not disable_pbar,
        num_sanity_val_steps=0, # avoid running validation on start
        log_every_n_steps=10
    )

    trainer.fit(pl_model, tr_loader, val_loader)
    trainer.test(pl_model, test_loader, verbose=False)


def main():
    # Timestamp for experiment
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # Set up command line args parser
    parser = argparse.ArgumentParser()
    
    # Method related
    parser.add_argument('--method', type=str, required=True,
            help='Method for training model. (mfvi, sl)')
    parser.add_argument('--params', type=str, required=False,
            help="Additional arguments for training algorithm")
    
    # Dataset related
    parser.add_argument('--dataset', type=str, required=True,
            help='Datasets to train eval.')
    parser.add_argument('--ds-params', type=str, required=False,
            help="Additional params for dataset in param1=val1,param2=val2,... format")
    parser.add_argument('--transform', type=str, required=False,
            default='normalize_x',
            help='Input transform to be applied. Defined in `transforms.py`')
    
    # Model related
    parser.add_argument('--model', type=str, required=True,
            help="Model to use. Options: LeNet")
    parser.add_argument('--model-params', type=str, required=False,
            help="Additional parameters to be passed to model.")
    
    # Optimization related
    parser.add_argument('--max-steps', type=int, required=False, 
            default=cfg.MAX_TRAIN_STEPS, 
            help="Number of steps to train the model (Default: {})".format(cfg.MAX_TRAIN_STEPS))
    parser.add_argument('--batch-size', type=int, required=False, 
            default=cfg.BATCH_SIZE_TR_LOADER,
            help="Minibatch size (Default: {})".format(cfg.BATCH_SIZE_TR_LOADER))

    # Loss related
    parser.add_argument('--wt-loss',  dest='wt_loss', action='store_true',
            help='Weighted loss function')
    parser.set_defaults(wt_loss=False)
    parser.add_argument('--mc-samples', type=int, required=False, default=32,
            help="Number of MonteCarlo forwards for averaging")
    
    # Others
    parser.add_argument('--outdir', type=str, required=False, default="zoo/test/",
            help="Parent output directory to save model")
    parser.add_argument('--gpu', dest='use_gpu', action='store_true')
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false')
    parser.add_argument('--prefix', type=str, required=False, default=None,
            help="Prefix for model directory")
    parser.set_defaults(use_gpu=True)
    parser.add_argument('--seed', type=int, required=False, default=None,
            help="Seed for running experiment")
    
    args = parser.parse_args()

    method = args.method
    method_params = args.params
    dataset = args.dataset
    ds_params = args.ds_params
    transform = args.transform
    model = args.model
    model_params = args.model_params
    max_steps = args.max_steps
    batch_size = args.batch_size
    wt_loss = args.wt_loss
    mc_samples = args.mc_samples
    use_gpu = args.use_gpu
    seed = args.seed

    # Parse any params
    ds_params = parse_params_str(ds_params)
    # base_params = parse_params_str(base_params)
    method_params = parse_params_str(method_params)
    model_params = parse_params_str(model_params)

    dataset_str = dataset 
    if ds_params:
        dataset_str += '-' + '-'.join([str(v) for _, v in ds_params.items()])

    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            args.outdir, 
                            dataset_str, 
                            model, 
                            args.prefix+"-"+timestamp if args.prefix else timestamp)

    
    # Print experiment configuration
    print("Method           :", method, method_params)
    print("Dataset          :", dataset, ds_params, transform)
    print("Model            :", model, model_params)
    print("Max steps        :", max_steps)
    print("Batch size       :", batch_size)
    print("Weighted Loss    :", wt_loss)
    print("MC samples       :", mc_samples)
    print("Use GPU          :", use_gpu)
    print("Outdir           :", outdir)

    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        print("INFO: Using seed `{}`.".format(seed))

    # Make output dir
    os.makedirs(outdir, exist_ok=True)

    # Writeout experiment configuration
    config_file_path = os.path.join(outdir, "config.json")
    with open(config_file_path, 'w') as fp:
        json.dump({
            'method': method,
            'method_params': method_params,
            'dataset': dataset,
            'ds_params': ds_params,
            'transform': transform,
            'model': model,
            'model_params': model_params,
            'max_steps': max_steps,
            'batch_size': batch_size,
            'wt_loss': wt_loss,
            'mc_samples': mc_samples,
            'seed': seed
        }, fp, indent=2)

    run_experiment(
        method, method_params,
        dataset, ds_params, transform, 
        model, model_params,
        max_steps, batch_size,
        wt_loss, 
        mc_samples, 
        use_gpu,
        outdir)

    print("") # New line at the end of experiment


if __name__=="__main__":
    main()
