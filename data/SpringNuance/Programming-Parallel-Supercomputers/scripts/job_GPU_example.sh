#!/bin/bash -l
#### For multi-GPU jobs, in --gpus-per-node N, choose N>1. Single GPU, N==1.
#### (Remember for Slurm #SBATCH, one # is active, two or
#### more ## is commented out)
####
#### If you have permanent Triton account (not made just for this
#### course), you can comment this out (comment = two `##`) and
#### maybe you will have shorter queue time.
#SBATCH --partition courses-gpu
#SBATCH --account courses
#### Standard parameters
#SBATCH --gpus-per-node 2
#SBATCH --time=00:05:00
#SBATCH --nodes 1             #Single node, for more nodes, increase
#SBATCH --ntasks-per-node 2   #Tasks per node, usually same as the gpus-per-node
##SBATCH --output=GPUcode.out

module purge   # unload all current modules
#For non-MPI job
module load openmpi
#For MPI job activate the below and de-activate the above
#module use /share/apps/scibuilder-spack/aalto-centos7-dev/2023-01/lmod/linux-centos7-x86_64/Core
#module load gcc/11.3.0 cmake/3.26.3 openmpi/4.1.5

nvcc -o GPUcode GPUcode.cu

time srun GPUcode
