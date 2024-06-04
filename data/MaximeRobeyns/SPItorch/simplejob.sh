#!/bin/bash

#SBATCH --job-name spitorch
#SBATCH --partition cnu
#SBATCH --ntasks 1
#SBATCH --gpus-per-task A100:1
#SBATCH --gpu-bind per_task:1
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-gpu 32G
#SBATCH --time 0-10:00:00
#SBATCH --output output.log

# Usage, for local runs:
# ./job.sh path/to/file.py +hydra=arg
# or for SLURM runs:
# sbatch job.sh path/to/file.py +hydra=arg

# For 80G A100s:
# #SBATCH --nodes 2
# SBATCH --nodelist bp1-gpu[038,039]
# #SBATCH --nodelist bp1-gpu039

# 1. Print environment information
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`

export SPS_HOME=$(pwd)/deps/fsps
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export NCCL_BLOCKING_WAIT=1

# 2. If running on BluePebble cluster, load appropriate modules to setup shell
if [[ $(hostname) == bp1-* ]]; then
    module load lang/python/anaconda/3.9.7-2021.12-tensorflow.2.7.0
    eval "$(conda shell.bash hook)"
    conda activate /user/work/`whoami`/condaenvs/spivenv
fi

python spt/modelling/simulation.py
python spt/inference/inference.py
