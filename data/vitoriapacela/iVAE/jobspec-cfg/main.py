# from comet_ml import Experiment
import argparse
import os
import pickle
import sys
import random

import numpy as np
import torch
import yaml

# from runners import ivae_runner, tcl_runner, ivae_sgd_runner, ivae_lbfgs_runner, ivae_data_runner, ivae_lbfgs2_runner, ivae_gd_runner
from runners import tcl_runner, ivae_lbfgs_runner, ivae_gd_runner, fastica_runner, ivae_adam_runner
# , ivae_lbfgs_batches_runner
torch.set_default_dtype(torch.double)
torch.backends.cudnn.benchmark = False
# torch.set_deterministic(True) # does not work

def parse():
    '''
    Config:
    Dataset descriptors
    nps: (int) number of points per segment (n_per_seg)
    ns: (int) number of segments (n_seg)
    dl: (int) dimension of latent sources (d_sources)
    dd: (int) d_data (dimension of the mixed data)
    nl: (int) number of layers for ICA mixing


    p: (str) probability distribution (e.g. 'gauss' for Normal, 'lap' for Laplace, 'hs' for Hypersecant)
    act: (str) activation function for the mixing transformation (e.g. 'none', 'lrelu', 'sigmoid')
    uncentered: (bool) if True, different distributions have different means
    noisy: (float) level of noise to add to the observations
    staircase: (bool) does not seem to be used.

    # model args
    n_layers: (int) number of layers in the MLP
    hidden_dim: (int) number of dimensions in each hidden layer
    activation: (str) activation function of the MLP (e.g. 'lrelu', 'none', 'sigmoid')
    ica: (bool) if True, run the iVAE. If False, run the VAE
    initialize: (bool) weight initialization? Does not seem to be active.
    batch_norm: (bool) batch normalization. Does not seem to be active.
    tcl: (bool) if True, run TCL. If False, run the iVAE

    # learning
    a: (int) weight of the logpx term of the ELBO
    b: (int) weight of the (logqs_cux - logqs) term of the ELBO
    c: (int) weight of the (logqs - logqs_i) term of the ELBO
    d: (int) weight of the (logqs_i - logps_cu) term of the ELBO
    (for standard iVAE loss: a, b, c, d = 1)
    gamma: (float?) if > 0, factor = True
    lr: (float) learning rate
    batch_size: (int) batch size
    epochs: (int) total number of epochs
    no_scheduler: (bool) if False, use a scheduler for the optimizer
    scheduler_tol: (int) scheduler tolerance
    anneal: (bool) annealing
    anneal_epoch: (int)

    # more configs
    shuffle: (bool) if True, shuffle data from the trainig batch
    one_hot: (bool) if True, one-hot encode the segments U
    checkpoint: (bool) if True, save the weights and meta-data in every epoch
    simple_mixing: if True, the elements of mixing matrix are sampled from U(-1,1)
    and mixing occurs through only one linear transformation.
    terms: (bool) if True, all the loss terms (logpx, logps_cu, logqs_cux) are saved during training.
    g: (bool) if True, the encoder (inference model) modeling the mean (g) is a simple MLP, equivalent to the unmixing matrix.
    discrete: (bool) if True, run discrete simulations (data, model) instead of continuous
    check_freq: (int) checkpoint frequency. If 1, saves after every epoch.
    identity: (bool) if True, use A=I as the mixing matrix for debugging purposes.
    early: (bool) if True, early stop.
    stop: (int) early stopping criteria. For example, if stop=20, stop training if the validation loss does not improve after 20 epochs.
    cond_thresh: (int) condition number threshold. Generate a new mixing matrix until its condition number is below this threshold.
    same_var: (bool) if True, generate segments with fixed variance = 1.
    simple_prior: (bool) if True, the prior models of mean and variance are simple MLPs, as opposed to a high-capacity neural network.

    -----
    args
    --s: (int) data generation seed.
    --seed: (int) learning seed.
    --run: (str) path containing the dataset, logs, and checkpoints. Do not use if you are checkpointing somewhere else.
    --doc: (str) creates another folder inside run/checkpoint, run/dataset, run/logs for documentation purpose. Do not use if you are checkpointing somewhere else.
    --n-sims: (int) number of simulations for each run. That is, number of learning seeds for each dataset.
    --m: (int) Bound of the uniform distribution to sample the means of the segments when generating data: U(-m, m).
    --dir_log: (str) logs directory. No need to change (and needs to be compatible with --run).
    --ckpt_folder: (str) checkpoints directory.
    '''
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--ns', type=int, help="Number of segments.", default=None)
    parser.add_argument('--custom_data_path', type=str, help="Custom data path.")
    parser.add_argument('--config', type=str, default='ivae.yaml', help='Path to the config file')
    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='', help='A string for documentation purpose')

    parser.add_argument('--n-sims', type=int, default=1, help='Number of simulations to run')
    parser.add_argument('--seed', type=int, default=0, help='Learning seed')
    parser.add_argument('--s', type=int, default=0, help='Data seed')
    parser.add_argument('--m', type=float, default=1, help='Mean bound')

    parser.add_argument('--dir_log', type=str, default='run/logs/', help='Logs directory')
    parser.add_argument('--ckpt_folder', type=str, default='/scratch/project_2002842/checkpoints/', help='Checkpoints directory')
    
    parser.add_argument('--init_noise', type=float, default=0.1, help='close_f weight initialization noise std')
    
    # Notice that boolean types do not behave as expected in the parser.
    # Use --feature, --no-feature, False is the default
    
    parser.add_argument('--set_f', dest='set_f', action='store_true', help='Fix mixing model weights to the true mixing matrix')
    parser.add_argument('--no-set_f', dest='set_f', action='store_false', help='Do NOT fix mixing model weights to the true mixing matrix')
    parser.set_defaults(set_f=False)
    
    parser.add_argument('--close_f', dest='close_f', action='store_true', help='Fix mixing model weights close the true mixing matrix')
    parser.add_argument('--no-close_f', dest='close_f', action='store_false', help='Do NOT fix mixing model weights close the true mixing matrix')
    parser.set_defaults(close_f=False)
    
    parser.add_argument('--set_prior', dest='set_prior', action='store_true', help='Fix prior model (mean and variance) weights to the true mean and variance values')
    parser.add_argument('--no-set_prior', dest='set_prior', action='store_false', help='Do NOT fix prior model (mean and variance) weights to the true mean and variance values')
    parser.set_defaults(set_prior=False)
    
    parser.add_argument('--fix_prior_mean', dest='fix_prior_mean', action='store_true', help='Fix prior means model weights to the true means')
    parser.add_argument('--no-fix_prior_mean', dest='fix_prior_mean', action='store_false', help='Do NOT fix prior means model weights to the true means')
    parser.set_defaults(set_prior=False)
    
    parser.add_argument('--fix_logl', dest='fix_logl', action='store_true', help='Fix prior log-variances model (logl) weights to the true log-variances')
    parser.add_argument('--no-fix_logl', dest='fix_logl', action='store_false', help='Do NOT fix prior log-variances model (logl) weights to the true log-variances')
    parser.set_defaults(set_prior=False)
    
    parser.add_argument('--set_inf', dest='set_inf', action='store_true', help='Fix latent sources to the true sources')
    parser.add_argument('--no-set_inf', dest='set_inf', action='store_false', help='Do NOT fix latent sources to the true sources')
    parser.set_defaults(set_inf=False)

    return parser.parse_args()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def make_dirs(args):
    os.makedirs(args.run, exist_ok=True)

    # the log directory holds .expid, which is used for checkpoints
    args.log = os.path.join(args.run, 'logs', args.doc)
    os.makedirs(args.log, exist_ok=True)

    args.checkpoints = os.path.join(args.run, 'checkpoints', args.doc)
    os.makedirs(args.checkpoints, exist_ok=True)

    args.data_path = os.path.join(args.run, 'datasets', args.doc)
    os.makedirs(args.data_path, exist_ok=True)


def main():
    args = parse()
    make_dirs(args)

    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # add device to config
    new_config = dict2namespace(config)
    new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED']=str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.set_default_dtype(torch.double)

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    
    if new_config.tcl:
        r = tcl_runner(args, new_config)
    elif new_config.fastica:
        r = fastica_runner(args, new_config)
    else:
#     elif new_config.ica: 
        if new_config.lbfgs:
            r = ivae_lbfgs_runner(args, new_config)
            # r = ivae_lbfgs_batches_runner(args, new_config)
        elif new_config.adam:
            r = ivae_adam_runner(args, new_config)
        else:
            r = ivae_gd_runner(args, new_config)
#         if new_config.custom_data:
#             r = ivae_data_runner(args, new_config)
#         elif new_config.lbfgs:
#             r = ivae_lbfgs_runner(args, new_config)
# #             r = ivae_lbfgs2_runner(args, new_config) #dont use
#         elif new_config.gd:
#             r = ivae_gd_runner(args, new_config)
#         else:
#             r = ivae_runner(args, new_config)
# #        r = ivae_sgd_runner(args, new_config) # dont use


if __name__ == '__main__':
    sys.exit(main())
