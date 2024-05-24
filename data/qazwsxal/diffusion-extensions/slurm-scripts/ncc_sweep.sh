#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1


#SBATCH -p "res-gpu-small"
#SBATCH --exclude="gpu[0-6]"
#SBATCH --qos="short"
#SBATCH -t 0-12:00:00

# Source the bash profile (required to use the module command)
source /etc/profile
module unload cuda
module load cuda/11.1

source .venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
# export PYTORCH_JIT=0
# export CUDA_LAUNCH_BLOCKING=1
wandb agent --count 1 "$@"