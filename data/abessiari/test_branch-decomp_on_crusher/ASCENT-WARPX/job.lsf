#!/bin/bash
# Begin LSF Directives
#SBATCH -A csc340
#SBATCH -t 02:00:00
#SBATCH -N 32
#SBATCH -J test-warpx
#SBATCH -o test-warpx.output
#SBATCH -e test-warpx.error


module load craype-accel-amd-gfx90a
module load rocm/5.1.0
module load cmake/3.22.1
module load gcc/11.2.0
module load git/2.31.1
module load git-lfs/2.11.0
module load cray-python/3.9.7.1
module load cray-mpich/8.1.15

# srun -n 192 --ntasks-per-node 6 -G 192 --gpus-per-node 6 ./warpx inputs_3d max_step=200 diag1.intervals=10 diag1.format=ascent
# nranks = 8 * 32
srun -n 256 --ntasks-per-node 8 --gpus-per-node 8 ./warpx inputs_3d max_step=200 diag1.intervals=10 diag1.format=ascent
