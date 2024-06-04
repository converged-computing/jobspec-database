#!/bin/bash

#SBATCH --job-name=KH-magnetization-1

#SBATCH --account=amath
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --gpus=6
#SBATCH --cpus-per-task=2
#SBATCH --array=1-@@NTASKS@@%6
#SBATCH --mem=20G
#SBATCH --time=48:00:00

#SBATCH --chdir=@@WORKDIR@@
#SBATCH --output=@@WORKDIR@@/sims/KH-%a/sim.log
#SBATCH --error=@@WORKDIR@@/sims/KH-%a/sim.err

module use ~/modulefiles
module load julia

export OPENBLAS_NUM_THREADS=1

julia --version
julia --project=../.. --startup-file=no main.jl --run $SLURM_ARRAY_TASK_ID
