import hddm
from copy import deepcopy
import pickle
import pandas as pd
import numpy as np
import pathlib
import psutil


import ssms

import arviz as az
from time import time 
import matplotlib
import matplotlib.pyplot as plt

import jax
from jax import numpy as  jnp

import math
import argparse
from time import time
import pathlib

from utils import save_traces

def traces_to_arviz_single_subject(traces = None, model = 'ddm'):
    traces_dict = {}
    
    for key_ in traces.keys():
        tmp_trace = traces[key_].values
        
        if '_trans' in key_:
            key_tmp = key_[:key_.find('_trans')]
            idx = hddm.model_config.model_config[model]['params'].index(key_tmp)
            a = hddm.model_config.model_config[model]['param_bounds'][0][idx]
            b = hddm.model_config.model_config[model]['param_bounds'][1][idx] #[exp(traces['z_trans'])
            traces_dict[key_tmp] = ((b - a) * np.exp(traces[key_].values) / (1 + np.exp(traces[key_].values))) + a
        else:
            traces_dict[key_] = traces[key_].values
    
    return az.from_dict(traces_dict), traces_dict

def traces_to_arviz_hierarchical(traces = None, model = 'ddm', n_subjects = 20):
    traces_dict = {}
    
    # clean up trans
    trans_columns = []
    for key_ in traces.keys():
        tmp_trace = traces[key_].values
        
        if '_trans' in key_:
            trans_columns.append(key_)
            
            if 'subj' in key_:
                key_tmp = key_[:key_.find('_trans')] + key_[(key_.find('_trans') + 6):]
            else:
                key_tmp = key_[:key_.find('_trans')]
            
            key_param_isolated = key_[:key_.find('_')]
            
            idx = hddm.model_config.model_config[model]['params'].index(key_param_isolated)
            a = hddm.model_config.model_config[model]['param_bounds'][0][idx]
            b = hddm.model_config.model_config[model]['param_bounds'][1][idx] #[exp(traces['z_trans'])
            traces[key_tmp] = ((b - a) * np.exp(traces[key_].values) / (1 + np.exp(traces[key_].values))) + a
    
    traces = traces.drop(trans_columns, axis = 1)
    
    # Deal with subject wise variables
    traces = traces.rename(columns = {param: param + '_mu_mu' for param in ssms.config.model_config[model]['params']},
                           inplace = False)
    traces = traces.rename(columns = {param + '_std': param + '_mu_std' for param in ssms.config.model_config[model]['params']},
                  inplace = False)
    
    for param in ssms.config.model_config[model]['params']:
        traces_dict[param + '_subj'] = traces[[param + '_subj.' + str(i) for i in range(n_subjects)]].values #.swapaxes(0,1)
        traces_dict[param + '_mu_mu'] = traces[param + '_mu_mu'].values
        traces_dict[param + '_mu_std'] = traces[param + '_mu_std'].values
    
    for key_ in traces_dict.keys():
        if traces_dict[key_].ndim > 1:
            traces_dict[key_] = np.expand_dims(traces_dict[key_], axis = 0)
            
    # Now turn to arviz inference data
    traces_inf = az.from_dict(traces_dict, 
                 coords = {param + \
                           '_subj_dim_0': np.arange(n_subjects) for param in ssms.config.model_config[model]['params']},
                 dims = {param + \
                         '_subj': [param + '_subj_dim_0'] for param in ssms.config.model_config[model]['params']},
                )
    
    traces_inf.add_groups({"posterior_predictive": traces_dict},
                          coords = {param + \
                           '_subj_dim_0': np.arange(n_subjects) for param in ssms.config.model_config[model]['params']},
                          dims = {param + \
                          '_subj': [param + '_subj_dim_0'] for param in ssms.config.model_config[model]['params']},
                         )
    
    return traces_inf, traces_dict

def stack_traces_dicts(traces_dicts = []):
    traces_dicts_stacked = {}
    for key_ in traces_dicts[0].keys():
        traces_dicts_stacked[key_] = np.stack([traces_dicts[i][key_] for i in range(len(traces_dicts))], axis = 0)
    return traces_dicts_stacked

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
    
    if args.modeltype == 'singlesubject':
        # Model
        model = args.model # for now only DDM (once we have choice probability models --> all models applicable)
        model_config = ssms.config.model_config[model].copy() # convenience

        central_data_single_subject = pickle.load(open('data/single_subject/' + model + \
                                                       '_nsamples_1000_nparams_200_stdfracdenom_6.pickle', 'rb'))

        for data_idx in range(args.idmin, args.idmax, 1):
            data = central_data_single_subject['data'][data_idx]['hddm']
            gt_params = central_data_single_subject['data'][data_idx]['gt_params']
            n_samples = central_data_single_subject['data'][data_idx]['hddm'].shape[0]

            # Make folder for arviz data if it doesn't already exist
            arviz_path = 'data/single_subject/' + \
                         model + '_nsamples_1000_nparams_200_stdfracdenom_6'

            pathlib.Path(arviz_path).mkdir(parents = True, 
                                           exist_ok = True)

            model_list = []
            start_t = time()

            for i in range(args.nchains):
                model_list.append(hddm.HDDMnn(data,
                                        model = model,
                                        include = hddm.model_config.model_config[model]['hddm_include']
                                 ))
                model_list[i].sample(args.nmcmc + args.nwarmup, burn = args.nwarmup)

            end_t = time()
            
            print('Inference took: ', end_t - start_t, ' seconds...')

            run_time = (end_t - start_t) / args.nchains

            traces_list = []
            traces_dict_list = []
            for i in range(2):
                traces_list.append(traces_to_arviz_single_subject(traces = model_list[i].get_traces(), 
                                                                 model = model)[0])
                traces_dict_list.append(traces_to_arviz_single_subject(traces = model_list[i].get_traces(),
                                                                       model = model)[1])
            mcmc_az = az.concat(traces_list, dim='chain')
            mcmc_az.posterior.attrs['runtime'] = run_time
            mcmc_az.posterior.attrs['machine_info'] = psutil.subprocess.run(['lscpu'], 
                                                                            capture_output=True, 
                                                                            text=True).stdout.split('\n')            
            traces_dicts_stacked = stack_traces_dicts(traces_dicts = traces_dict_list)

            print('saving files...')
            save_traces(file_path = arviz_path,
                        arviz_trace = mcmc_az,
                        dict_trace = traces_dicts_stacked,
                        model = args.model,
                        idx = data_idx,
                        backend = 'hddm',
                        infer_type = 'mcmc',                        
                        machine = args.machine,
                        nparticles = args.nchains)
        
    elif args.modeltype == 'hierarchical':
        model = args.model
        model_config = ssms.config.model_config[args.model]

        central_data_hierarchical = pickle.load(open('data/hierarchical/' + model + \
                                                     '_nsamples_1000_nsubjects_20_nparams_200' + \
                                                     '_stdfracdenom_6.pickle', 'rb'))

        for data_idx in range(args.idmin, args.idmax, 1):
            data = central_data_hierarchical['data'][data_idx]['hddm']
            gt_params = central_data_hierarchical['data'][data_idx]['gt_params']
            n_samples = central_data_hierarchical['data'][data_idx]['hddm'].shape[0]
            n_subjects = central_data_hierarchical['data'][data_idx]['hddm'].shape[1]


            # Make folder for arviz data if it doesn't already exist
            arviz_path = 'data/hierarchical/' + \
                         model + '_nsamples_1000_nsubjects_20_nparams_200_stdfracdenom_6'

            pathlib.Path(arviz_path).mkdir(parents = True, 
                                           exist_ok = True)

            model_list = []
            start_t = time()

            for i in range(args.nchains):
                model_list.append(hddm.HDDMnn(data,
                                              model = model,
                                              include = hddm.model_config.model_config[model]['hddm_include'],
                                              is_group_model = True,
                                              p_outlier = 0.0
                                              ))
                model_list[i].sample(args.nmcmc + args.nwarmup, burn = args.nwarmup)

            end_t = time()

            print('Inference took: ', end_t - start_t, ' seconds...')
            
            run_time = (end_t - start_t) / args.nchains

            traces_list = []
            traces_dict_list = []

            for i in range(2):
                traces_list.append(traces_to_arviz_hierarchical(traces = model_list[i].get_traces(), 
                                                                model = model,
                                                                n_subjects = n_subjects)[0])
                traces_dict_list.append(traces_to_arviz_hierarchical(traces = model_list[i].get_traces(),
                                                                     model = model,
                                                                     n_subjects = n_subjects)[1])
            mcmc_az = az.concat(traces_list, dim = 'chain')
            mcmc_az.posterior.attrs['runtime'] = run_time
            mcmc_az.posterior.attrs['machine_info'] = psutil.subprocess.run(['lscpu'], 
                                                                            capture_output=True, 
                                                                            text=True).stdout.split('\n')             
            traces_dicts_stacked = stack_traces_dicts(traces_dicts=traces_dict_list)
            
            print('saving files...')
            save_traces(file_path = arviz_path,
                        arviz_trace = mcmc_az,
                        dict_trace = traces_dicts_stacked,
                        model = args.model,
                        idx = data_idx,
                        backend = 'hddm',
                        infer_type = 'mcmc',
                        machine = args.machine,
                        nparticles = args.nchains)