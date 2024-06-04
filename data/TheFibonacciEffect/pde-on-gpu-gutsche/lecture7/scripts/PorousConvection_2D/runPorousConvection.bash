#!/bin/bash -l
#SBATCH --job-name="PC_2D_daint"
#SBATCH --output=PC_2D_daint.%j.o
#SBATCH --error=PC_2D_daint.%j.e
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account class04


module load daint-gpu
module load Julia/1.7.2-CrayGNU-21.09-cuda
srun julia -O3 --check-bounds=no --project=../.. ./PorousConvection_2D_xpu_daint.jl 511 1023 4000