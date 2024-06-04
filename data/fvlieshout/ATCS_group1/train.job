#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=roberta_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=1:00:00
#SBATCH --mem=32000M
#SBATCH --array=1
#SBATCH --output=logs/job_outputs/roberta_%A_%a.out

module purge
module load 2019
module load Python/3.7.5-foss-2019b
module load CUDA/10.1.243
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module load Anaconda3/2018.12

# Your job starts in the directory where you call sbatch
cd $HOME/ATCS_group1

# Activate your environment
source activate gnn-env

PARAMETERS_FILE=parameters.txt

srun python -u train.py $(head -$SLURM_ARRAY_TASK_ID $PARAMETERS_FILE | tail -1)
