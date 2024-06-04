#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:1      
#SBATCH --cpus-per-task=2           
#SBATCH --mem=32G     
               
module load python/3.9 cuda/11.7 

# Loading your environment
source ~/venvs/gflownet/bin/activate

python ~/gflownet/src/gflownet/tasks/main.py