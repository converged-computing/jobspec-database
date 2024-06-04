#!/bin/bash

#SBATCH -c 1
#SBATCH -t 1-04:00                  # wall time (D-HH:MM)
#SBATCH -p general -q public
#SBATCH -G 1
#SBATCH --mem=50G
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)

module load mvapich2-2.3.7-gcc-11.2.0
MV2_USE_ALIGNED_ALLOC=1
# export JULIA_CUDA_MEMORY_POOL=none
time mpirun -n 1 julia --project --check-bounds=no -O3 /scratch/tpatel28/topo_mag/EW_sim/IGG_gpu/main_temporal.jl 
# time julia --project --check-bounds=no -O3 /scratch/tpatel28/topo_mag/EW_sim/main.jl 
# time python vtkgenscript.py