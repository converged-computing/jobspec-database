import torch
import pyro
import pyro.distributions as dist
import matplotlib
import matplotlib.pyplot as plt
from torch.distributions.constraints import positive

import logging
import os
import psutil

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import scipy

import pyro
from pyro.infer import MCMC, NUTS
import ssms
import lanfactory
torch.set_default_dtype(torch.float32)

from lanfactory.trainers.torch_mlp import TorchMLP 

import lanfactory
import ssms

import arviz as az
import cloudpickle as cpickle
#from bokeh.resources import INLINE
#import bokeh.io
#from bokeh import *
#bokeh.io.output_notebook(INLINE)

import argparse
import pathlib
from time import time
from copy import deepcopy

# Necessary?
pyro.enable_validation(True)
pyro.set_rng_seed(20)
logging.basicConfig(format='%(message)s', level=logging.INFO)


network_files = {'network_config': {'ddm': 'd27193a4153011ecb76ca0423f39a3e6_ddm_torch__network_config.pickle',
                                    'angle': 'eba53550128911ec9fef3cecef056d26_angle_torch__network_config.pickle',
                                    'weibull': '44deb16a127f11eca325a0423f39b436_weibull_torch__network_config.pickle',
                                    'levy': '80dec298152e11ec88b8ac1f6bfea5a4_levy_torch__network_config.pickle'},
                 'network': {'ddm': 'd27193a4153011ecb76ca0423f39a3e6_ddm_torch_state_dict.pt',
                             'angle': 'eba53550128911ec9fef3cecef056d26_angle_torch_state_dict.pt',
                             'weibull': '44deb16a127f11eca325a0423f39b436_weibull_torch_state_dict.pt',
                             'levy': '80dec298152e11ec88b8ac1f6bfea5a4_levy_torch_state_dict.pt'}
                }

initial_params_dict = {'ddm': {'v': torch.tensor(0.5),
                               'a': torch.tensor(1.5),
                               'z': torch.tensor(0.5),
                               't': torch.tensor(1.0)
                              },
                       'angle': {'v': torch.tensor(0.5),
                                 'a': torch.tensor(1.5),
                                 'z': torch.tensor(0.5),
                                 't': torch.tensor(1.0),
                                 'theta': torch.tensor(0.5)
                                },
                       'weibull': {'v': torch.tensor(0.5),
                                   'a': torch.tensor(1.5),
                                   'z': torch.tensor(0.5),
                                   't': torch.tensor(1.0),
                                   'theta': torch.tensor(0.5)
                                  },
                       'levy': {'v': torch.tensor(0.5),
                                'a': torch.tensor(1.5),
                                'z': torch.tensor(0.5),
                                't': torch.tensor(1.0),
                                'alpha': torch.tensor(1.5),
                               },
                      }

def load_network(model = 'ddm', machine = 'gpu'):
    network_config = pickle.load(open('nets/' + network_files['network_config'][model],
                                 'rb'))
    network = lanfactory.trainers.torch_mlp.TorchMLP(network_config = network_config,
                                                 input_shape = model_config['n_params'] + 2)
    
    if machine == 'cpu':
        network.load_state_dict(torch.load('nets/' + network_files['network'][model], 
                                           map_location = torch.device('cpu')))
    else:
        network.load_state_dict(torch.load('nets/' + network_files['network'][model]))

    network.eval()
    return network

import math
from numbers import Real
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all

from utils import save_traces

class SSMDist(dist.TorchDistribution):
    def __init__(self, params, num_trials, network, ssm_model):
        self.net = network
        self.num_trials = num_trials
        self.ssm_model = ssm_model
        self.params = params
        
        if self.params.dim() == 1:
            batch_shape = torch.Size((1,))
        elif self.params.dim() > 1:
            batch_shape = self.params.size()[:-1]
            
        super().__init__(batch_shape = batch_shape, 
                         event_shape = torch.Size((2, )), 
                         validate_args=False)
        
    def sample(self):
        return sim_wrap(theta = params, 
                        model = self.ssm_model)
    
    def log_prob(self, value):
        if self.params.dim() == 1:
            tmp_params = self.params.tile(dims = (self.num_trials, 1))
            tmp_dat = value
        
        if self.params.dim() == 3:
            tmp_params = self.params.repeat_interleave(self.num_trials, -2)
            tmp_dat = value.repeat((self.params.size()[0], 1, 1))
        
        net_in = torch.cat([tmp_params, tmp_dat], 
                           dim = -1)
        
        # Old
        logp = torch.clip(self.net(net_in), min = -16.11).squeeze(dim = -1)
        return logp
    
class SSMDistH(dist.TorchDistribution):
#     arg_constraints = {'loc': constraints.interval(-1, 1),
#                        'scale': constraints.interval(0.0001, 10)
#                       }
    def __init__(self, params, num_trials, network, ssm_name):
        self.net = network
        self.n_samples = num_trials
        self.boundaries = ssms.config.model_config[ssm_name]['param_bounds']
        self.out_of_bounds_val = -66.1
        self.params = params
        self.ssm_name = ssm_name
        self.params_shape = self.params.size()
        batch_shape = self.params.size()[:-1]
        
        super().__init__(batch_shape = batch_shape, #torch.Size((my_size,)),
                         event_shape = torch.Size((2,)), 
                         validate_args = False) #torch.Size((2,))) # event_shape = (1,))
        
    def sample(self):
        return sim_wrap(theta = self.params, model = self.ssm_name, n_samples = self.n_samples)
    
    def log_prob(self, value):
        
        if self.params.dim() == 3:
            tmp_dat = value.repeat((self.params_shape[0], 1, 1, 1))
            tmp_params = self.params.unsqueeze(1).tile((1, self.n_samples, 1, 1))         
        elif self.params.dim() == 4:
            tmp_dat = value.repeat(self.params_shape[0], self.params_shape[1], 1, 1, 1)
            tmp_params = self.params.unsqueeze(2).tile((1, 1, self.n_samples, 1, 1))
        else:
            tmp_dat = value
            tmp_params = self.params.tile((self.n_samples, 1, 1))
        
        net_in = torch.cat([tmp_params, tmp_dat], dim = -1)
        
        logp = torch.clip(self.net(net_in), min = -16.11)
        logp_squeezed = torch.squeeze(logp, dim = -1)
        
        for i in range(ssms.config.model_config[self.ssm_name]['n_params']):
            logp_squeezed = torch.where(net_in[..., i] < torch.tensor(self.boundaries[1][i]), 
                                        logp_squeezed, 
                                        torch.tensor(self.out_of_bounds_val))
            logp_squeezed = torch.where(net_in[..., i] > torch.tensor(self.boundaries[0][i]), 
                                        logp_squeezed, 
                                        torch.tensor(self.out_of_bounds_val))
        
        logp_squeezed = torch.squeeze(logp_squeezed) #, dim = 0) #.unsqueeze(1)
        return logp_squeezed

def model_maker(model = 'ddm'):
    model_config = ssms.config.model_config[model]
    def ssm_model(num_trials, data, network, model):
        param_list = []
        for param in model_config['params']:
            idx = model_config['params'].index(param)
            param_list.append(pyro.sample(param, dist.Uniform(model_config['param_bounds'][0][idx],
                                                              model_config['param_bounds'][1][idx])))

        with pyro.plate("data", num_trials) as data_plate:
            return pyro.sample("obs", 
                               SSMDist(torch.stack(param_list, dim = -1), 
                                       num_trials, 
                                       network, 
                                       model), 
                               obs = data)
    return ssm_model

def model_maker_hierarchical(model = 'ddm'):
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
            mu_mu_list.append(pyro.sample(param + "_mu_mu", 
                                         dist.Normal(param_mean, 
                                                     model_config['mu_mu_std'][model][param])
                                        )
                             )
            mu_std_list.append(pyro.sample(param + "_mu_std",
                                          dist.HalfNormal(100.)
                                         )
                              )           
                              
        with pyro.plate("subjects", num_subjects) as subjects_plate:
            subj_list = []
            
            for param in model_config['params']:
                idx = model_config['params'].index(param)
                subj_list.append(pyro.sample(param + "_subj", dist.Normal(mu_mu_list[idx], mu_std_list[idx])
                                           )
                                )
            with pyro.plate("data", num_trials) as data_plate:
                return pyro.sample("obs",
                                  SSMDistH(torch.stack(subj_list, axis = -1), num_trials, network, ssm_name),
                                  obs = data)
    return ssm_model_hierarchical

from torch.distributions import transform_to
from typing import Callable, Optional

def init_to_uniform(
    site: Optional[dict] = None,
    radius: float = 2.0,
    ):
    """
    Initialize to a random point in the area ``(-radius, radius)`` of
    unconstrained domain.

    :param float radius: specifies the range to draw an initial point in the
        unconstrained domain.
    """
    if site is None:
        return functools.partial(init_to_uniform, radius=radius)

    value = site["fn"].sample().detach()
    t = transform_to(site["fn"].support)
    value = t(torch.rand_like(t.inv(value)) * (2 * radius) - radius)
    value._pyro_custom_init = False
    return value

def init_to_uniform_custom(
    site: Optional[dict] = None,
    radius: float = 0.1,
    ):
    """
    Initialize to a random point in the area ``(-radius, radius)`` of
    unconstrained domain.

    :param float radius: specifies the range to draw an initial point in the
        unconstrained domain.
    """
    if site is None:
        return functools.partial(init_to_uniform, radius=radius)
    
    site_num = site['fn'].shape()
    # group level locs -------
    if site['name'] == 'v_mu_mu':
        return torch.rand(1) - 0.5
    
    if site['name'] == 'a_mu_mu':
        return (torch.rand(1) - 0.5) + 1.5
    
    if site['name'] == 'z_mu_mu':
        return (torch.rand(1) / 5) + 0.3
    
    if site['name'] == 't_mu_mu':
        return torch.rand(1) / 5
    
    if site['name'] == 'theta_mu_mu':
        return (torch.rand(1) / 10)
    
    if site['name'] == 'alpha_mu_mu':
        return (torch.rand(1) + 1.5)
    
    if site['name'] == 'beta_mu_mu':
        return (torch.rand(1) + 1.5)
    # -------
    
    # subject level locs -------
    if site['name'] == 'v_subj':
        return torch.rand(site_num) - 0.5  
    
    if site['name'] == 'a_subj':
        return (torch.rand(site_num) - 0.5) + 1.5
    
    if site['name'] == 'z_subj':
        return (torch.rand(site_num) / 5) + 0.3
        
    if site['name'] == 't_subj':
        return torch.rand(site_num) / 5
    
    if site['name'] == 'theta_subj':
        return (torch.rand(site_num) / 10)
    
    if site['name'] == 'alpha_subj':
        return (torch.rand(site_num) + 1.5)
    
    if site['name'] == 'beta_subj':
        return (torch.rand(site_num) + 1.5)
    
    # -------
    
    # std parameters keep the uniform random init
    value = site["fn"].sample().detach()
    t = transform_to(site["fn"].support)
    value = t(torch.rand_like(t.inv(value)) * (2 * radius) - radius)
    value._pyro_custom_init = False
    return value

if __name__ == "__main__":
    # Command line interface arguments
    CLI = argparse.ArgumentParser()
    
    CLI.add_argument("--model",
                     type = str,
                     default = "ddm")
    CLI.add_argument("--modeltype",
                     type = str,
                     default = "singlesubject")
    CLI.add_argument("--nsteps",
                     type = int,
                     default = 5000)
    CLI.add_argument("--nparticles",
                     type = int,
                     default = 10)
    CLI.add_argument("--guide",
                     type = str,
                     default = 'normal')
    CLI.add_argument("--machine",
                     type = str,
                     default = 'gpu')
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
    
    # Model
    if args.modeltype == 'singlesubject':
        model = args.model # for now only DDM (once we have choice probability models --> all models applicable)
        model_config = ssms.config.model_config[model].copy() # convenience

        central_data_single_subject = pickle.load(open('data/single_subject/' + model + \
                                                       '_nsamples_1000_nparams_200_stdfracdenom_6.pickle', 'rb'))
        
        arviz_path = 'data/single_subject/' + \
                         model + '_nsamples_1000_nparams_200_stdfracdenom_6'

        pathlib.Path(arviz_path).mkdir(parents = True, 
                                       exist_ok = True)


        for data_idx in range(args.idmin, args.idmax, 1):
            data = central_data_single_subject['data'][data_idx]['pyro']
            gt_params = central_data_single_subject['data'][data_idx]['gt_params']
            n_samples = central_data_single_subject['data'][data_idx]['pyro'].shape[0]
            network = load_network(model = model,
                                   machine = args.machine)

            ssm_model = model_maker(model = model)

            pyro.clear_param_store()

            # These should be reset each training loop.
            if args.guide == 'normal':
                auto_guide = pyro.infer.autoguide.AutoNormal(ssm_model)
            elif args.guide == 'mvnormal':
                auto_guide = pyro.infer.autoguide.AutoMultivariateNormal(ssm_model)
            
            adam = pyro.optim.Adam({"lr": 0.02})
            elbo = pyro.infer.Trace_ELBO(num_particles = args.nparticles, 
                                         vectorize_particles = True)
            svi = pyro.infer.SVI(ssm_model, auto_guide, adam, elbo)

            losses = []
            start_t = time()
            for step in range(args.nsteps):  # Consider running for more steps.
                loss = svi.step(*(n_samples, data, network, model))
                losses.append(loss)
                if step % 100 == 0:
                    logging.info("Step {} out of {}, Elbo loss: {}".format(step, args.nsteps, loss))
            end_t = time()
            
            print('Inference took: ', end_t - start_t, ' seconds...')

            predictive = pyro.infer.Predictive(ssm_model, guide = auto_guide, num_samples = 2000)
            svi_samples = predictive(*(n_samples, data, network, model)) #torch.tensor(data))

            svi_samples_new = {key_: svi_samples[key_].numpy().swapaxes(0, 1) for key_ in svi_samples.keys() if key_ != 'obs'}

            az_svi = az.convert_to_inference_data(svi_samples_new)
            az_svi.posterior.attrs['runtime'] = end_t - start_t
            az_svi.posterior.attrs['svi_steps'] = args.nsteps
            az_svi.posterior.attrs['svi_loss_trajectory'] = losses
            az_svi.posterior.attrs['machine_info'] = psutil.subprocess.run(['lscpu'], 
                                                                            capture_output=True, 
                                                                            text=True).stdout.split('\n')
            
            # Make folder for arviz data if it doesn't already exist
            print('saving files...')
            save_traces(file_path = arviz_path,
                        arviz_trace = az_svi,
                        dict_trace = svi_samples_new,
                        model = args.model,
                        idx = data_idx,
                        backend = 'pyro',
                        infer_type = 'variational',
                        machine = args.machine,
                        nparticles = args.nparticles)
            
    
    elif args.modeltype == 'hierarchical':
        model = args.model
        model_config = ssms.config.model_config[model]

        central_data_hierarchical = pickle.load(open('data/hierarchical/' + model + \
                                                '_nsamples_1000_nsubjects_20_nparams_200_stdfracdenom_6.pickle', 
                                                'rb'))

        for data_idx in range(args.idmin, args.idmax, 1):
            data = central_data_hierarchical['data'][data_idx]['pyro']
            gt_params = central_data_hierarchical['data'][data_idx]['gt_params']
            n_samples = central_data_hierarchical['data'][data_idx]['pyro'].shape[0]
            n_subjects = central_data_hierarchical['data'][data_idx]['pyro'].shape[1]
            network = load_network(model = model, 
                                   machine = args.machine)

            ssm_model_hierarchical = model_maker_hierarchical(model = model)

            pyro.clear_param_store()

            # These should be reset each training loop.
            if args.guide == 'normal':
                auto_guide_hierarchical = pyro.infer.autoguide.AutoNormal(ssm_model_hierarchical, 
                                                                          init_scale = 0.02, 
                                                                          init_loc_fn = init_to_uniform_custom)
            elif args.guide == 'mvnormal':
                auto_guide_hierarchical = pyro.infer.autoguide.AutoMultivariateNormal(ssm_model_hierarchical, 
                                                                                      init_scale = 0.02, 
                                                                                      init_loc_fn = init_to_uniform_custom)
                
            adam = pyro.optim.Adam({"lr": 0.02})
            elbo = pyro.infer.Trace_ELBO(num_particles = args.nparticles, vectorize_particles = True)
            svi = pyro.infer.SVI(ssm_model_hierarchical, auto_guide_hierarchical, adam, elbo)

            losses = []
            start_t = time()

            # Main loop
            for step in range(args.nsteps): 
                loss = svi.step(*(n_subjects, n_samples, data, network, model))
                losses.append(loss)
                if step % 100 == 0:
                    logging.info("Step {} out of {}, Elbo loss: {}".format(step, args.nsteps, loss))
            end_t = time()

            print('Inference took: ', end_t - start_t, ' seconds...')

            # plt.figure(figsize=(5, 2))
            # plt.plot(losses)
            # plt.xlabel("SVI step");
            # plt.ylabel("ELBO loss");
            
            # Predictive
            predictive = pyro.infer.Predictive(ssm_model_hierarchical, guide = auto_guide_hierarchical, num_samples = 2000)
            svi_samples = predictive(n_subjects, n_samples, data, network, model) #torch.tensor(data))

            # Get ready for arviz
            svi_samples_new = {key_: svi_samples[key_].numpy().swapaxes(0, 1) for key_ in svi_samples.keys() if key_ != 'obs'}

            # Make into inference data
            az_svi = az.from_dict(posterior = svi_samples_new, 
                                  posterior_predictive = {'obs': np.expand_dims(svi_samples['obs'].numpy(), 0)})
            az_svi.posterior.attrs['runtime'] = end_t - start_t
            az_svi.posterior.attrs['svi_steps'] = args.nsteps
            az_svi.posterior.attrs['svi_loss_trajectory'] = losses
            az_svi.posterior.attrs['machine_info'] = psutil.subprocess.run(['lscpu'], 
                                                                            capture_output=True, 
                                                                            text=True).stdout.split('\n')
            
            # Make folder for arviz data if it doesn't already exist            
            arviz_path = 'data/hierarchical/' + \
                         model + '_nsamples_1000_nsubjects_20_nparams_200_stdfracdenom_6'

            pathlib.Path(arviz_path).mkdir(parents = True, 
                                           exist_ok = True)
            
            print('saving files...')
            save_traces(file_path = arviz_path,
                        arviz_trace = az_svi,
                        dict_trace = svi_samples_new,
                        model = args.model,
                        idx = data_idx,
                        backend = 'pyro',
                        infer_type = 'variational',
                        machine = args.machine,
                        nparticles = args.nparticles)