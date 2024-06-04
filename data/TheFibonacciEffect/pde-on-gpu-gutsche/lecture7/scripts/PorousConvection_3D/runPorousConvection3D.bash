#!/bin/bash -l
#SBATCH --job-name="PC_3D 127 2000 false true"
#SBATCH --output=PC_3D.%j.o
#SBATCH --error=PC_3D.%j.e
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account class04

module load daint-gpu
module load Julia/1.7.2-CrayGNU-21.09-cuda
srun julia -O3 --check-bounds=no --project=../.. ./PorousConvection_3D_xpu.jl
# srun julia -O3 --check-bounds=no --project=../.. ./PorousConvection_3D_xpu.jl 127 100 false true
# nz = 127,nt= 2000,do_vis=false,save_arr=true