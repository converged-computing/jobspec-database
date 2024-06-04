#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=ExampleJob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:30:00
#SBATCH --mem=48000M
#SBATCH --output=slurm_output_%A.out

#module purge
#module load 2019
#module load Python/3.7.5-foss-2019b
#module load CUDA/10.1.243
#module load cuDNN/7.6.5.32-CUDA-10.1.243
#module load NCCL/2.5.6-CUDA-10.1.243
#module load Anaconda3/2018.12
#conda env create -f environment.yml


# Your job starts in the directory where you call sbatch
cd $HOME/ATCS/group_assignment
# Activate your environment
source activate python385
# Run your code
srun python -u classify_emotion.py
#classify_emotion.py
