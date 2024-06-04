#!/bin/bash -l
#SBATCH --job-name="strong_scaling"
#SBATCH --output=strong_scaling.%j.o
#SBATCH --error=strong_scaling.%j.e
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account class04


module load daint-gpu
module load Julia/1.7.2-CrayGNU-21.09-cuda
srun julia -O3 --check-bounds=no --project=../../.. l8_diffusion_2D_pref_multixpu._SC.jl