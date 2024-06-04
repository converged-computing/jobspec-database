#!/bin/bash

#SBATCH --partition=long
#SBATCH -c 6                                                           
#SBATCH --mem=32G                                        
#SBATCH --time=48:00:00    
#SBATCH --gres=gpu:1                           
#SBATCH -o /home/mila/l/luke.rowe/qflowoffline/slurm_logs/iql_hopper_medium-expert-v2_seed0.out
module --quiet load miniconda/3
conda activate qflow_new
wandb login $WANDB_API_KEY
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mila/l/luke.rowe/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CPATH=$CONDA_PREFIX/include
cd IQL_PyTorch
python main.py --track --env-name hopper-medium-expert-v2 --seed 0