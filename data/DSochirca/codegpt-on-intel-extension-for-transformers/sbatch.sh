#!/bin/bash

#SBATCH --job-name=xtc
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-courses-cse3000

# Change to your project name along with above slurm info
PROJECT_NAME="xtc"

# Load relevant modules for env creation
module load 2022r2
module load miniconda3/4.12.0
module load cuda/11.7

# Set caching env variables
export MPLCONFIGDIR="./envs/$PROJECT_NAME/.cache/matplotlib/"
export HF_DATASETS_CACHE="./envs/$PROJECT_NAME/.cache/huggingface/"

# Store current GPU usage
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

# Create environment 
conda env create --file environment.yml --name $PROJECT_NAME
conda activate $PROJECT_NAME

# Run project (model training)
srun python CodeCompletion.py > code_completion_$PROJECT_NAME.log

# Get GPU usage 
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
