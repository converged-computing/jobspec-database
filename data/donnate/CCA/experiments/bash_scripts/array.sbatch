#!/bin/bash

#SBATCH --job-name=array-job
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --array=1-100
#SBATCH --time=35:00:00
#SBATCH --account=pi-cdonnat
#SBATCH --ntasks=1
#SBATCH --partition=caslake
#SBATCH --mem=20G

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# Add lines here to run your computations.
module load python
module load pytorch

cd $SCRATCH/$USER/CCA
echo $1
echo $2
python3 experiments/experiment.py --model $1 --epochs 2000 --patience 3 --dataset $2 --lr $3 --normalize $4 --result_file $SLURM_ARRAY_TASK_ID
~
~
