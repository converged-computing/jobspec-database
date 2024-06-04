#!/bin/bash

#SBATCH -t 05:30:00
#SBATCH -p normal_q
#SBATCH -N 1
#SBATCH -A ravt
# Job name:
#SBATCH --job-name=mcmc_runner
#SBATCH --output=R-%x.%j.out

module load Python

./venv/bin/python plot_maps_helper.py --states $1 --start $SLURM_ARRAY_TASK_ID
