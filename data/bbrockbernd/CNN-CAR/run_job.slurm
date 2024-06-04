#!/bin/bash -l
#
#SBATCH --job-name="dist_training"
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G
#SBATCH --gres=gpu:4


module load 2022r1
module load gpu
module load python/3.8.12-bohr45d
module load openmpi
module load py-tensorflow




srun python Model2D.py
