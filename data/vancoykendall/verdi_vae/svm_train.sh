#!/bin/bash
#SBATCH -o job_logs/svm_train.sh.log-%j-%a
#SBATCH -c 5
#SBATCH -a 1-6
echo test1
# Loading the required module
module load anaconda/2020b
echo test2
# Load conda environment
# conda activate disentangling-vae
echo test3
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Number of Tasks: " $SLURM_ARRAY_TASK_COUNT

# Run script
python svm_models.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT