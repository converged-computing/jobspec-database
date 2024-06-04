#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1

module load opencl-intel/16.4 
module load opencl-nvidia/9.0
module load openmpi/gcc/64/1.10.3

./vexcl.out "$@"
