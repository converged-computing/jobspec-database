#!/bin/bash

#SBATCH --account=PHYS030544
#SBATCH --job-name=ll_mpi
#SBATCH --partition=teach_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=100M
#SBATCH --time=0:05:00

echo 'running mpi test'

module load languages/miniconda

echo "Before activation: $(which python)"

#/mnt/storage/software/languages/miniconda/miniconda3.11.3-CompressAI-Vision0.1.3/condabin/conda activate desktop_clone_env
#conda init bash
#possibly I need to run "conda init bash" every time i open a new terminal
source activate desktop_clone_env

echo "After activation: $(which python)"

#module load languages/intel/2020-u4

srun --mpi=pmi2 python LebwohlLasher_mpi.py 50 50 0.5 0

