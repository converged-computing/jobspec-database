#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=Socrat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=02:00:00
#SBATCH --mem=32000M
#SBATCH --output=slurm_output_%A.out

module purge
module load 2021
module load Anaconda3/2021.05

# Activate your environment
source activate socrat
# Run your code
srun python -u main.py --data_dir $TMPDIR/ --lm 'google/flan-t5-xl'
