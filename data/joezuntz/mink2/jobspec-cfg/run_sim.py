import numpy as np
import os
import random
import sys
sys.path.append("./simulation_code/")
from simulate_des_maps import *

os.environ["PATH"]='/home/ngrewal/flask/bin:'+os.environ["PATH"] 

# simulation parameters
nside = 1024
smoothing = 0

# fiducial parameter values
omega_b = 0.048
omega_m = 0.3
h = 0.7
n_s = 0.96
sigma_8 = 0.8
b1 = 1.42
b2 = 1.65
b3 = 1.60
b4 = 1.92
b5 = 2.00

# instead of rank use slurm job id (will this work across processes)
index = int(os.environ['SLURM_ARRAY_TASK_ID'])

# where to save simulations
path = '/disk01/ngrewal/Fiducial_Simulations'

# function that generates n/p number of simulations
def run_sim(seed):
    #random.seed(seed)
    c_maps,l_maps = simulate_des_maps_bias(omega_b, omega_m, h, n_s, sigma_8, b1, b2, b3, b4, b5, smoothing, nside, seed)
    np.save(os.path.join(path, f'cmaps_{index}'),c_maps)
    np.save(os.path.join(path, f'lmaps_{index}'),l_maps)

# the bit that will run in each process
run_sim(index)  # use random seed so each process generates different maps
