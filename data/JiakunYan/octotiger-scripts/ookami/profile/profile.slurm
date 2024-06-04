#!/bin/bash
#SBATCH --partition=large
#SBATCH --time=5:00
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=profile
#SBATCH --output=slurm_output.%x-o%j
#SBATCH --error=slurm_error.%x-o%j

task="rs"
pp="lci"
max_level="6"
mode=${1:-"stat"}

srun --mpi=pmix bash -x ${ROOT_PATH}/profile_wrapper.sh $task $pp $max_level $mode