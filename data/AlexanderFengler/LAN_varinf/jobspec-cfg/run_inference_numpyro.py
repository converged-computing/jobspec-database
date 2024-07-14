import numpy as np
import lanfactory
from lanfactory.trainers.torch_mlp import LoadTorchMLPInfer
import ssms
import pandas as pd
import pickle
from copy import deepcopy
import psutil

import torch
from lanfactory.trainers.torch_mlp import TorchMLP  

import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'    

import arviz as az
from time import time
import scipy

import matplotlib
import matplotlib.pyplot as plt

import jax
#jax.config.update('jax_platform_name', 'cpu')
from jax import numpy as jnp

import math
from numbers import Real
from numbers import Number

import numpyro as npy
from numpyro import distributions as dist
from numpyro.distributions import constraints

import ssms

from numbers import Real
from numbers import Number

from numpyro.infer import MCMC, NUTS
from jax import random, vmap

import argparse
import pathlib
from time import time

from utils import save_traces

network_files = {'network_config': {'ddm': 'd27193a4153011ecb76ca0423f39a3e6_ddm_torch__network_config.pickle',
                                    'angle': 'eba53550128911ec9fef3cecef056d26_angle_torch__network_config.pickle',
                                    'weibull': '44deb16a127f11eca325a0423f39b436_weibull_torch__network_config.pickle',
                                    'levy': '80dec298152e11ec88b8ac1f6bfea5a4_levy_torch__network_config.pickle'},
                 'network': {'ddm': 'd27193a4153011ecb76ca0423f39a3e6_ddm_torch_state_dict.pt',
                             'angle': 'eba53550128911ec9fef3cecef056d26_angle_torch_state_dict.pt',
                             'weibull': '44deb16a127f11eca325a0423f39b436_weibull_torch_state_dict.pt',
                             'levy': '80dec298152e11ec88b8ac1f6bfea5a4_levy_torch_state_dict.pt'}
                }

initial_params_jax_dict =  {'ddm': {'v': jnp.array([2.0, 2.0]),
                                    'a': jnp.array([1.0, 1.0]),
                                    'z': jnp.array([0.5, 0.5]),
                                    't': jnp.array([1.0, 1.0])
                                   },
                            'angle': {'v': jnp.array([0.5, 0.5]),
                                      'a': jnp.array([1.5, 1.5]),
                                      'z': jnp.array([0.5, 0.5]),
                                      't': jnp.array([1.0, 1.0]),
                                      'theta': jnp.array([0.5, 0.5])
                                     },
                            'weibull': {'v': jnp.array([0.5, 0.5]),
                                        'a': jnp.array([1.5, 1.5]),
                                        'z': jnp.array([0.5, 0.5]),
                                        't': jnp.array([1.0, 1.0]),
                                        'alpha': jnp.array([2.0, 2.0]),
                                        'beta': jnp.array([2.0, 2.0]),
                                       },
                            'levy': {'v': jnp.array([0.5, 0.5]),
                                     'a': jnp.array([1.5, 1.5]),
                                     'z': jnp.array([0.5, 0.5]),
                                     't': jnp.array([1.0, 1.0]),
                                     'alpha': jnp.array([1.5, 1.5]),
                                    }
                           }

class CustomTorchMLP:
    def __init__(self, state_dict, network_config):
        self.weights = []
        self.jnp_weights = []
        self.biases = []
        self.jnp_biases = []
        self.activations = deepcopy(network_config['activations'])
        self.net_depth = len(self.activations)
        self.state_dict = state_dict
        
        self.activation_dict_jax = {'tanh': jnp.tanh}
        self.activation_dict_torch = {'tanh': torch.tanh}
        cnt = 0
        
        for obj in self.state_dict:
            if 'weight' in obj:
                self.weights.append(deepcopy(self.state_dict[obj]).T)
                self.jnp_weights.append(jnp.asarray(self.weights[-1].numpy()))
            elif 'bias' in obj:
                self.biases.append(torch.unsqueeze(deepcopy(self.state_dict[obj]), 0))
                self.jnp_biases.append(jnp.asarray(self.biases[-1].numpy()))
                
    def forward(self, input_tensor):
        tmp = input_tensor
        for i in range(0, self.net_depth - 1, 1):
            tmp = torch.tanh(torch.add(torch.matmul(tmp, self.weights[i]), self.biases[i]))
        tmp = torch.add(torch.matmul(tmp, self.weights[self.net_depth - 1]), self.biases[self.net_depth - 1])
        return tmp
    
    def forward_jnp(self, input_array):
        tmp = input_array
        for i in range(0, self.net_depth - 1, 1):
            #tmp = jnp.tanh(jnp.dot(tmp, self.jnp_weights[i]) + (self.jnp_biases[i]))
            tmp = self.activation_dict_jax[self.activations[i]](jnp.dot(tmp, self.jnp_weights[i]) + (self.jnp_biases[i]))
        tmp = jnp.dot(tmp, self.jnp_weights[self.net_depth - 1]) + self.jnp_biases[self.net_depth - 1]
        return tmp
    
def load_network(model = 'ddm'):
    model_config = deepcopy(ssms.config.model_config[model])
    
    # Load network config
    network_config = pickle.load(open('nets/' + network_files['network_config'][model],
                                      'rb'))

    # Initialize network class
    torch_net = TorchMLP(network_config = network_config,
                   input_shape = model_config['n_params'] + 2,
                   generative_model_id = None)

    # Load weights and biases
    torch_net.load_state_dict(torch.load('nets/' + network_files['network'][model],
                              map_location=torch.device('cpu')))


    # Initialize custom pytorch network
    custom_torch_net = CustomTorchMLP(torch_net.state_dict(), 
                                      network_config)
    
    return custom_torch_net       
        
class SSMDistJax(npy.distributions.Distribution):
    def __init__(self, params, network):
        self.net = network
        self.params = params
        
        if self.params.ndim == 1:
             batch_shape = (1,)
        elif self.params.ndim > 1:
            batch_shape = self.params.shape[:-1]
            
        super().__init__(batch_shape = batch_shape, event_shape = (2,))
        
    def sample(self, key, sample_shape = ()):
        raise NotImplementedError
    
    def log_prob(self, value):
        net_in = jnp.hstack([jnp.tile(self.params, ((value.shape[0], 1))), value])
        net_out = self.net.forward_jnp(net_in)
        out = jnp.squeeze(jnp.clip(net_out, a_min = -16.11))
        return out
    
class SSMDistHJax(npy.distributions.Distribution):
    def __init__(self, params, num_trials, network, ssm_name):
        self.net = network
        self.n_samples = num_trials
        self.boundaries = ssms.config.model_config[ssm_name]['param_bounds']
        self.out_of_bounds_val = -66.1
        self.params = params
        self.params_size = self.params.shape
        self.n_params = ssms.config.model_config[ssm_name]['n_params']
        self.ssm_name = ssm_name
        self.params_shape = self.params.shape 
        super().__init__(batch_shape = self.params_shape[:-1], 
                         event_shape = (2,)) #torch.Size((2,))) # event_shape = (1,))
        
    def sample(self):
        raise NotImplementedError
    
    def log_prob(self, value):
        if self.params.ndim == 3:
            tmp_dat = jnp.tile(value, reps = (self.params_size[0], 1, 1, 1))
            tmp_params = jnp.tile(self.params, reps = (1, self.n_samples, 1, 1))
        else:
            tmp_params = jnp.tile(self.params, reps = (self.n_samples, 1, 1))
            tmp_dat = value

        net_in = jnp.concatenate([tmp_params, tmp_dat], axis = -1)
        net_out = self.net.forward_jnp(net_in)
        logp_squeezed = jnp.squeeze(jnp.clip(net_out, a_min = -16.11))

        for i in range(self.n_params):
            logp_squeezed = jnp.where(net_in[..., i] < self.boundaries[1][i], 
                                      logp_squeezed, 
                                      self.out_of_bounds_val)
            logp_squeezed = jnp.where(net_in[..., i] > self.boundaries[0][i], 
                                      logp_squeezed, 
                                      self.out_of_bounds_val)
        return logp_squeezed
   
def model_maker_jax(model = 'ddm'):
    model_config = ssms.config.model_config[model]
    
    def ssm_model_jax(num_trials, data, network):
        param_list = []
        for param in model_config['params']:
            idx = model_config['params'].index(param)
            param_list.append(npy.sample(param, dist.Uniform(model_config['param_bounds'][0][idx],
                                                             model_config['param_bounds'][1][idx])))
        with npy.plate("data", num_trials) as data_plate:
            return npy.sample("obs", SSMDistJax(jnp.stack(param_list, axis=-1), network), obs = data)
        
    return ssm_model_jax

def model_maker_hierarchical_jax(model = 'ddm'):
    model_config = deepcopy(ssms.config.model_config[model])
    model_config['mu_mu_std'] = {'ddm': {'v': 0.5,
                                         'a': 0.5,
                                         'z': 0.5,
                                         't': 0.5
                                        },
                                 'angle':{'v': 0.5,
                                          'a': 0.5,
                                          'z': 0.5,
                                          't': 0.5,
                                          'theta': 0.5
                                         },
                                 'weibull':{'v': 0.5,
                                            'a': 0.5,
                                            'z': 0.5,
                                            't': 0.5,
                                            'alpha': 0.5,
                                            'beta': 0.5
                                           },
                                 'levy': {'v': 0.5,
                                          'a': 0.5, 
                                          'z': 0.5,
                                          't': 0.5,
                                          'alpha': 0.5
                                         }
                                }
    def ssm_model_hierarchical(num_subjects, num_trials, data, network, ssm_name):
        mu_mu_list = []
        mu_std_list = []
        for param in model_config['params']:
            idx = model_config['params'].index(param)
            param_mean = (model_config['param_bounds'][1][idx] + model_config['param_bounds'][0][idx]) / 2
            mu_mu_list.append(npy.sample(param + "_mu_mu", 
                                         dist.Normal(param_mean, 
                                                     model_config['mu_mu_std'][model][param])
                                        )
                             )
            mu_std_list.append(npy.sample(param + "_mu_std",
                                          dist.HalfNormal(100.)
                                         )
                              )
                              
                              
        with npy.plate("subjects", num_subjects) as subjects_plate:
            subj_list = []
            
            for param in model_config['params']:
                idx = model_config['params'].index(param)
                subj_list.append(npy.sample(param + "_subj", dist.Normal(mu_mu_list[idx], mu_std_list[idx])
                                           )
                                )
            with npy.plate("data", num_trials) as data_plate:
                return npy.sample("obs",
                                  SSMDistHJax(jnp.stack(subj_list, axis = -1), num_trials, network, ssm_name),
                                  obs = data)
    return ssm_model_hierarchical    

if __name__ == "__main__":
    # Command line interface arguments
    CLI = argparse.ArgumentParser()
    
    CLI.add_argument("--model",
                     type = str,
                     default = "ddm")
    CLI.add_argument("--modeltype",
                     type = str,
                     default = "singlesubject")
    CLI.add_argument("--machine",
                     type = str,
                     default = 'cpu')
    CLI.add_argument("--nchains",
                     type = int,
                     default = 2)
    CLI.add_argument("--nwarmup",
                     type = int,
                     default = 2000)
    CLI.add_argument("--nmcmc",
                     type = int,
                     default = 3000)
    CLI.add_argument("--idmin",
                     type = int,
                     default = 0)
    CLI.add_argument("--idmax",
                     type = int,
                     default = 100)
    CLI.add_argument("--progressbar",
                     type = int,
                     default = 0)
    
    args = CLI.parse_args()
    print(args)
    
    model = args.model # for now only DDM (once we have choice probability models --> all models applicable)
    model_config = ssms.config.model_config[model].copy() # convenience

    if args.modeltype == 'singlesubject':
        # Load rt/choice dataset
        central_data_single_subject = pickle.load(open('data/single_subject/' + model + \
                                                       '_nsamples_1000_nparams_200_stdfracdenom_6.pickle', 'rb'))

        for data_idx in range(args.idmin, args.idmax, 1):
            # Load basic data
            data = central_data_single_subject['data'][data_idx]['numpyro']
            gt_params = central_data_single_subject['data'][data_idx]['gt_params']
            n_samples = central_data_single_subject['data'][data_idx]['numpyro'].shape[0]
            network = load_network(model = model)
            
            # Load numpyro probabilistic model
            numpyro_ssm_model = model_maker_jax(model = model)

            # Run NUTS
            nuts_kernel = NUTS(numpyro_ssm_model)
            
            start_t = time()
            mcmc = MCMC(nuts_kernel, 
                        num_samples = args.nmcmc, 
                        num_warmup = args.nwarmup, 
                        num_chains = args.nchains,
                        progress_bar = args.progressbar)  
            rng_key = random.PRNGKey(20)
            rng_key, rng_key_ = random.split(rng_key)
            mcmc.run(rng_key_, n_samples, data, network, init_params = initial_params_jax_dict[model])
            end_t = time()
            
            print('Inference took: ', end_t - start_t, ' seconds...')

            # Convert samples to useful formats
            mcmc_az = az.from_numpyro(mcmc)
            mcmc_az.posterior.attrs['runtime'] = end_t - start_t
            mcmc_az.posterior.attrs['machine_info'] = psutil.subprocess.run(['lscpu'], 
                                                                            capture_output=True, 
                                                                            text=True).stdout.split('\n')
            traces_dict = mcmc.get_samples(group_by_chain = True)

            # Save inference files
            arviz_path = 'data/single_subject/' + \
                         model + '_nsamples_1000_nparams_200_stdfracdenom_6'
            
            print('saving files...')
            save_traces(file_path = arviz_path,
                        arviz_trace = mcmc_az,
                        dict_trace = traces_dict,
                        model = args.model,
                        idx = data_idx,
                        backend = 'numpyro',
                        infer_type = 'mcmc',
                        machine = args.machine,
                        nparticles = args.nchains)
            
    if args.modeltype == 'hierarchical':
        # Load rt/choice dataset
        central_data_hierarchical = pickle.load(open('data/hierarchical/' + model + \
                                                     '_nsamples_1000_nsubjects_20_nparams_200_stdfracdenom_6.pickle', 'rb'))

        for data_idx in range(args.idmin, args.idmax, 1):
            # Load basic data
            data = central_data_hierarchical['data'][data_idx]['numpyro']
            gt_params = central_data_hierarchical['data'][data_idx]['gt_params']
            n_samples = central_data_hierarchical['data'][data_idx]['numpyro'].shape[0]
            n_subjects = central_data_hierarchical['data'][data_idx]['numpyro'].shape[1]
            network = load_network(model = model)

            # Load numpy probabilistic model
            ssm_model_hierarchical = model_maker_hierarchical_jax(model = model)

            #Run NUTS
            init_params_dict_auto = {key_: jnp.stack([gt_params[key_] for j in range(args.nchains)], axis = 0) for key_ in gt_params.keys()}
            nuts_kernel = NUTS(ssm_model_hierarchical)
            
            start_t = time()
            mcmc = MCMC(nuts_kernel, 
                        num_samples = args.nmcmc, 
                        num_warmup = args.nwarmup, 
                        num_chains = args.nchains,
                        progress_bar = args.progressbar)  

            rng_key = random.PRNGKey(20)
            rng_key, rng_key_ = random.split(rng_key)

            mcmc.run(rng_key_, 
                     *(n_subjects, n_samples, data, network, 'ddm'), # model parameter
                     init_params = init_params_dict_auto # other parameters
                    )
            end_t = time()
            
            print('Inference took: ', end_t - start_t, ' seconds...')

            mcmc_az = az.from_numpyro(mcmc)
            mcmc_az.posterior.attrs['runtime'] = end_t - start_t
            mcmc_az.posterior.attrs['machine_info'] = psutil.subprocess.run(['lscpu'], 
                                                                            capture_output=True, 
                                                                            text=True).stdout.split('\n')
            traces_dict = mcmc.get_samples(group_by_chain = True)

            # Make folder for arviz data if it doesn't already exist
            arviz_path = 'data/hierarchical/' + \
                         model + '_nsamples_1000_nsubjects_20_nparams_200_stdfracdenom_6'

            print('saving files...')
            save_traces(file_path = arviz_path,
                        arviz_trace = mcmc_az,
                        dict_trace = traces_dict,
                        model = args.model,
                        idx = data_idx,
                        backend = 'numpyro',
                        infer_type = 'mcmc',
                        machine = args.machine,
                        nparticles = args.nchains)