#!/bin/bash

#SBATCH --job-name=RT-magnetization-3

#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2
#SBATCH --array=1-@@NTASKS@@%2
#SBATCH --mem=20G
#SBATCH --time=120:00:00

#SBATCH --chdir=@@WORKDIR@@
#SBATCH --output=@@WORKDIR@@/sims/RT-%a/sim.log
#SBATCH --error=@@WORKDIR@@/sims/RT-%a/sim.err

module use ~/modulefiles
module load julia

export OPENBLAS_NUM_THREADS=1

julia --version
julia --project=../.. --startup-file=no main.jl --run $SLURM_ARRAY_TASK_ID
