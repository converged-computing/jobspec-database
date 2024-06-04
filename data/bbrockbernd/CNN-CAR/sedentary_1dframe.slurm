#!/bin/bash -l
#
#SBATCH --job-name="1d_sedentary"
#SBATCH --time=23:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=96G
#SBATCH --gres=gpu:1


module load 2022r1
module load gpu
module load python/3.8.12-bohr45d
module load openmpi
module load py-tensorflow




srun python Model1D2S_SEDENTARY.py
