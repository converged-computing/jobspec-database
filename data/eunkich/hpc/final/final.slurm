#!/bin/bash

#SBATCH --job-name=final
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eunkich@uw.edu

#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=5G
#SBATCH --gpus=1
#SBATCH --time=00-00:15:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --export=all
#SBATCH --output=slurm.out # where STDOUT goes
#SBATCH --error=slurm.err # where STDERR goes

# Modules to use (optional).
# <e.g., module load singularity>

module load cuda

rm *.csv

# Your programs to run.
source build_oblas_cublas.sh;
source build_mem_swaps.sh;
source build_file_swaps.sh;
source build_cpu_gpu_bw.sh;
source build_fftw.sh;
source build_cufft.sh;

rm *.o x*