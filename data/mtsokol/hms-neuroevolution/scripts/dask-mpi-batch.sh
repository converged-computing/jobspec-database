#!/bin/bash -l

#SBATCH -J hms-neuro-job
#SBATCH -p plgrid-short
#SBATCH -A <GRANT_ID>
#SBATCH --ntasks-per-node=24
#SBATCH --mem=55G
#SBATCH -t 01:00:00
#SBATCH --nodes=10

export NOISE_PATH=/net/archive/groups/plgghmsneuro/noise.npy
module load plgrid/libs/python-mpi4py/3.0.1-python-3.6
mpirun -np 240 python3 -m implementation.experiments.hms_atari_sea -e 60