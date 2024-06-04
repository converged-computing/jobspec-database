#!/bin/bash
#SBATCH --account=geo
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -J dask_host
#SBATCH --time=06:00:00
#SBATCH --mail-user=jbusecke@princeton.edu
#SBATCH --mail-type=ALL

# SLURM Template for Princeton Tiger
# Scheduler: SLURM

# This writes a scheduler.json file into your home directory
# You can then connect with the following Python code
# >>> from dask.distributed import Client
# >>> client = Client(scheduler_file='~/scheduler.json')

DASKDIR=~/.dask_tmp

rm -r $DASKDIR/worker*
source activate standard
export XDG_RUNTIME_DIR=""
rm -f scheduler.json
mpirun --np 4 dask-mpi --nthreads 4 --memory-limit 'auto' --bokeh-port 7771 --interface ib0 --local-directory $DASKDIR

