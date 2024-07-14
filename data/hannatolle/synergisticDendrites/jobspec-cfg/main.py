from itertools import product
import sys
import simulations_dendritic_dyn
from time import time

# stop the time
start = time()

# Define indices of input synapses (indices of 2 out of 193 dendrite sections)
params = [(i, j) for i, j in product(range(193), repeat=2) if i != j]

# Fetch job_id from command-line argument and set parameters
path2home = sys.argv[1]
job_id = int(sys.argv[2])
nn = params[job_id-1]

# Run the simulation and save results
simulations_dendritic_dyn.run(nn, path2home)

# give output
stop = time()
print(f'finished job {job_id} with sections {nn[0]} and {nn[1]}.')
print(f'elapsed time: {stop-start}')
