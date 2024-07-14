from __future__ import division
import os
import json
import random

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import multiprocessing as mp
np.set_printoptions(suppress=True, precision=5)

from src.gaussianMixture import *

nprocs = 25

# Load config and create output directories
config_path = "configs/ksc.json"
data_path = "data/simulated/sbc"

with open(config_path) as f:
    config = json.load(f)

# Create output directories
output_dir = "simulation_output/" + config["simulation_name"] 
os.mkdir(output_dir)
os.mkdir(output_dir + "/output")
os.mkdir(output_dir + "/tmp")
os.mkdir(output_dir + "/std_out")

# Save config
with open(output_dir + "/config.json", "w") as outfile:
    json.dump(config, outfile, indent=4)

# Set seed and configure seed, data location and chains
random.seed(config["seed"])

sample_vect = random.sample(list(range(1, int(1e6))), config["n_iterations"] * config["chains"])
data_file_indexes = [i for i in range(1, config["n_iterations"]+1) for _ in range(config["chains"])]
seed_data_files = list(zip(sample_vect, data_file_indexes))


def simulation(seed_data):
    random.seed(seed_data[0])
    data_file = data_path + "/" + config["data_location"] + f"/{seed_data[1]}.json"

    with open(data_file) as f:
        prior_params = json.load(f)

    endog = pd.DataFrame(dict(y_obs = prior_params["y_sim"]))["y_obs"]

    # Setup the model and simulation smoother
    mod = TVLLDT(endog)
    mod.set_smoother_output(0, smoother_state=True) # Calculate and return the smoothed states. Other options like cov matrices available
    sim = mod.simulation_smoother()

    # Storage for traces
    trace_smoothed = np.zeros((config["mcmc_samples"] + 1, mod.nobs))
    trace_states = np.zeros((config["mcmc_samples"] + 1, mod.nobs))
    trace_mixing = np.zeros((config["mcmc_samples"] + 1, mod.nobs), int)
    trace_mu = np.zeros((config["mcmc_samples"] + 1, 1))
    trace_phi = np.zeros((config["mcmc_samples"] + 1, 1))
    trace_sigma2 = np.zeros((config["mcmc_samples"] + 1, 1))

    # Initial values (p. 367)
    trace_mixing[0] = 0 # Initialised all with the 4th gaussian. So all draws from a specific normal distribution. Not specified in the paper how this is done
    trace_mu[0] = 0
    trace_phi[0] = 0.95
    trace_sigma2[0] = 0.02 #0.5 in the code. says 0.02 in the paper

    # Iterations
    for s in range(1, config["mcmc_samples"] + 1):
        # Update the parameters of the model
        params = np.r_[trace_mu[s-1], trace_phi[s-1], trace_sigma2[s-1]]
        mod.update_mixing(indicators = trace_mixing[s-1], 
                          params = params,
                          parameterisation = config["parameterisation"])
                          
        mod.update(params = params, 
                   parameterisation = config["parameterisation"],
                   transformed=True)

        # Simulation smoothing
        sim.simulate()
        states = sim.simulated_state
        trace_states[s] = states[0]

        # Draw mixing indicators
        trace_mixing[s] = draw_mixing(mod, states)

        # Draw parameters
        trace_phi[s] = draw_posterior_phi(mod, states, trace_phi[s-1], trace_mu[s-1], trace_sigma2[s-1])
        trace_sigma2[s] = draw_posterior_sigma2(mod, states, trace_phi[s-1], trace_mu[s-1])
        trace_mu[s] = draw_posterior_mu(mod, states, trace_phi[s-1], trace_sigma2[s-1])

    # Create dataframe
    y_star = np.log(endog**2 + 0.001)
    weights = importance_weights(data = y_star, 
          states = trace_states, 
          burn = config["burn"])

    burn_draws = np.concatenate((trace_mu, trace_phi, trace_sigma2, trace_states), axis = 1)
    param_draws = burn_draws[config["burn"]+1:burn_draws.shape[0]]
    draws = np.concatenate((param_draws, weights.reshape(-1,1)), axis = 1)

    if config['reweight_samples']:
        n_samples = weights.shape[0]
        resampled_indexes = np.random.choice(np.arange(0,n_samples), size=n_samples, replace=True, p=weights)
        resampled_draws = draws[resampled_indexes]
        samples = pd.DataFrame(resampled_draws)
    else:
        samples = pd.DataFrame(draws)
    
    static_names = ['mu', 'phi', 'sigma2']
    state_names = [f"h[{state}]" for state in np.arange(1, trace_states.shape[1]+1)]
    samples.columns = static_names + state_names + ['weights']

    # Save as parquet
    table = pa.Table.from_pandas(samples)
    pq.write_table(table, output_dir + "/tmp/" + f'{seed_data[1]}_{seed_data[0]}.parquet')

# parallelise
# import multiprocessing as mp
# print("Number of processors: ", mp.cpu_count())
# Parallel(n_jobs=2)(delayed(simulation)(i) for i in seed_data_files[0:2])

def slice_data(data, nprocs):
    aver, res = divmod(len(data), nprocs)
    nums = []
    for proc in range(nprocs):
        if proc < res:
            nums.append(aver + 1)
        else:
            nums.append(aver)
    count = 0
    slices = []
    for proc in range(nprocs):
        slices.append(data[count: count+nums[proc]])
        count += nums[proc]
    return slices

pool = mp.Pool(processes=nprocs)
inp_lists = slice_data(seed_data_files, nprocs)
result = [pool.map_async(simulation, inp) for inp in inp_lists]
[x for p in result for x in p.get()]
