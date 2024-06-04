#!/bin/bash
#SBATCH --job-name=my_dask_job
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1000
#SBATCH --time=00:40:00
#SBATCH --partition=regular2

module purge
module load gnu8/8.3.0
module load openmpi3/3.1.4

source /home/znazari/.bashrc

conda activate Zainab-env
mpiexec --version

mpiexec -n 16 python  hi.py
