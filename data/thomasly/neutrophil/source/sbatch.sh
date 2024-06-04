#!/bin/bash

#SBATCH --job-name=tensorflow
##SBATCH --nodes=1
##SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu4_medium
##SBATCH --error=/gpfs/home/liuy08/slurm_outputs/%x_%j.err 
##SBATCH --output=/gpfs/home/liuy08/slurm_outputs/%x_%j.out
##SBATCH --dependency=afterany:job_id

module purge
module load python/gpu/3.6.5
wait

python train_resnet.py 1 32 100 0

