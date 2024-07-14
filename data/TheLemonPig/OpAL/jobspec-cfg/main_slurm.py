from simulator import Simulator
from analysis import state_heatmap, success_metrics, plot_trends, auc
from simulation_config import config_

import argparse
import pandas as pd
from datetime import datetime
import pickle
import numpy as np
from itertools import product
import os

def grid_search(config, seed: int):
    np.random.seed(seed)
    hyperparams = config['hyperparams']
    lists_of_hyperparams = dict()
    for k,v in hyperparams.items():
        if type(v) == tuple:
            start, stop, step = v
            stop += 1e-10
            sublist = np.arange(start, stop, step)
        elif type(v) == list:
            sublist = v
        lists_of_hyperparams.update({k: np.round(sublist,decimals=5)})
    # lists_of_hyperparams = {k: np.round(np.arange(*v),decimals=5) for k, v in hyperparams.items()}
    param_permutations = list(product(*lists_of_hyperparams.values()))
    models = config['model_params']
    environments = config['environment_params']
    meta_results = []
    new_params = param_permutations[-1]
    for idx, params in enumerate(param_permutations):
        if idx % 10 == 0:
            print(f'\n{idx} out of {len(param_permutations)}\n')
        new_params = {k: np.round(params[idx], 5) for idx, k in enumerate(hyperparams.keys())}
        for param in new_params.keys():
            for i, model in enumerate(models):
                if param == 'alpha' and model['model'].startswith("OpAL"):
                    model[param+'_g'] = new_params[param]
                    model[param + '_n'] = new_params[param]
                elif param in model.keys():
                    model[param] = new_params[param]

                config_['model_params'][i] = model
        print(new_params)
        simulator = Simulator(config['environment_params'], config['model_params'])
        results = simulator.run(reps=1, steps=config['epochs'], seed=seed)
        meta_results.append(results)
    prefix = 'OpAL_' if any([model['model'].startswith('OpAL' ) for model in models]) else ''
    filepath = os.path.join(os.path.join(os.getcwd(), f'{prefix}Logs'), f'{prefix}Config.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(config, f)
    # filepath = os.path.join(os.path.join(os.getcwd(), f'{prefix}Logs'), f'{prefix}Results_{seed}.pkl')
    # with open(filepath, 'wb') as f:
    #     pickle.dump(meta_results, f)
    n_permutations = len(param_permutations)
    n_models = len(models)
    n_envs = len(environments)
    n_contexts = n_models * n_envs
    success_rates = np.zeros((n_permutations, n_contexts))
    for idx, results in enumerate(meta_results):
        success_rate = success_metrics(config, results, 1, test_ratio=0, verbose=False)
        success_rates[idx] += np.array(success_rate)
    success_rates = np.round(success_rates,decimals=5)
    savepath = os.path.join(os.path.join(os.getcwd(), 'Data'), f'{prefix}Success_Rates_{seed}.csv')
    df = pd.DataFrame({
        f"{models[n%n_models]['name']}/{environments[n//n_models]['name']}": success_rates[:,n] for n in range(n_contexts)
    })
    df.to_csv(savepath, index=False)

    test_rates = np.zeros((n_permutations, n_contexts))
    for idx, results in enumerate(meta_results):
        test_rate = success_metrics(config, results, 1, test_ratio=0.1, verbose=False)
        test_rates[idx] += np.array(test_rate)
    test_rates = np.round(test_rates,decimals=5)
    savepath = os.path.join(os.path.join(os.getcwd(), 'Data'), f'{prefix}Test_Rates_{seed}.csv')
    df = pd.DataFrame({
        f"{models[n%n_models]['name']}/{environments[n//n_models]['name']}": test_rates[:,n] for n in range(n_contexts)
    })
    df.to_csv(savepath, index=False)

    aucs = np.zeros((n_permutations, n_contexts))
    for idx, results in enumerate(meta_results):
        auc_results = auc(config, results, 1, verbose=False)
        aucs[idx] += np.array(auc_results)
    aucs = np.round(aucs,decimals=5)
    savepath = os.path.join(os.path.join(os.getcwd(), 'Data'), f'{prefix}AUCs_{seed}.csv')
    df = pd.DataFrame({
        f"{models[n%n_models]['name']}/{environments[n//n_models]['name']}": aucs[:,n] for n in range(n_contexts)
    })
    df.to_csv(savepath, index=False)
    return n_permutations

parser = argparse.ArgumentParser("parser")
parser.add_argument('--slurm_id', type=int, help='number of slurm array')
args = parser.parse_args()
rep = args.slurm_id
# rep = 0
start = datetime.now()
n_permutations = grid_search(config_, seed=rep)
end = datetime.now()
print(f'Number of parameter permutations tested: {n_permutations}')
n_models = len(config_["model_params"])
n_envs = len(config_["environment_params"])
print(f'Number of Models: {n_models}   Number of Envs: {n_envs}')
n_epochs = config_["epochs"]
print(f'Total Epochs: {n_epochs}')
n_minutes = (end-start).total_seconds()/60
print(f'Duration: {n_minutes} minutes')
print(f'Duration per model: {n_minutes / (n_models)}')
print(f'Duration per context: {n_minutes / (n_models * n_envs)}')
print(f'Duration per run: {n_minutes / (n_models * n_envs * n_permutations)}')