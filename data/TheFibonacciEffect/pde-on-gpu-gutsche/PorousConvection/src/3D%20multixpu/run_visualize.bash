#!/bin/bash -l
#SBATCH --job-name="viz_3D_porous_convection"
#SBATCH --output=viz_3D_porous_convection.%j.o
#SBATCH --error=viz_3D_porous_convection.%j.e
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account class04

module load daint-gpu
module load Julia/1.7.2-CrayGNU-21.09-cuda
srun julia --project=../.. ./visualise.jl