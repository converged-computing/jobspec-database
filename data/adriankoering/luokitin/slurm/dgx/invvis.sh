#!/bin/bash
#SBATCH --job-name=extra_rgb_rn34
#SBATCH --partition=p2
#SBATCH --time=1:00:00


# DGX features 10 threads and 62 GB memory per GPU (6.25 GB per CPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=6G

# mkdir -p logs/slogs/${SLURM_SLURM_ARRAY_JOB_ID}
#SBATCH -o logs/slogs/%A_%a.out
#SBATCH -e logs/slogs/%A_%a.err

##SBATCH --export ALL # exports env-varialbes from current shell to job?

#SBATCH --nice=1000
#SBATCH --array=1-24%2

export WANDB_PROJECT=invvis

# Set these values per environment variable rather than config
# s.t. we are reminded to update them before launching a run
export WANDB_NAME=${SLURM_JOB_NAME}
export WANDB_JOB_TYPE=devel
export WANDB_JOB_NAME=${WANDB_NAME}
export WANDB_TAGS="devel,rgb,extra,resnet34"
# export WANDB_MODE=disabled

export HYDRA_FULL_ERROR=1 
srun python train.py experiment=invvis/rgb dataset.data_dir=/home/koering/data/invvis/webds/extra model/encoder=resnet34 # -c job --resolve