#!/bin/bash

#SBATCH -N 1                        # number of compute nodes
#SBATCH -c 1                        # number of tasks your job will spawn
#SBATCH --mem=0                    # amount of RAM requested in GiB (2^40)
#SBATCH -p general -q public
#SBATCH -G a100:4                # Request two GPUs
#SBATCH -t 0-20:00                  # wall time (D-HH:MM)
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)

MV2_USE_ALIGNED_ALLOC=1
module load mvapich2-2.3.7-gcc-11.2.0
time mpirun -n 4 julia rk45_main_multiple.jl
module unload mvapich2-2.3.7-gcc-11.2.0
