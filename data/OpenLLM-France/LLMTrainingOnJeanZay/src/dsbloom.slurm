#!/bin/bash
#SBATCH --job-name=ds_bloom
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:50:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --cpus-per-task=8
#SBATCH --account=knb@a100
#SBATCH -C a100

# hack to avoid issues of very small $HOME
export HOME=$WORK"/home/"

# env with latest deepspeed, transformers, datasets
. $HOME/envs/ds/bin/activate

# to get idr_torch
module load cpuarch/amd
module load pytorch-gpu/py3/2.2.0

## launch script on every node
set -x

echo "DATEDEBUT"
date

srun python dsbloom.py

