#!/bin/bash

#SBATCH --job-name=CVHW1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=5:00
#SBATCH --mem=1GB
#SBATCH --output=wordcounts_%A_%a.out
#SBATCH --error=wordcounts_%A_%a.err
module purge
module load python/intel/3.8.6
cd /scratch/$USER/myjarraytest
python wordcount.py sample-$SLURM_ARRAY_TASK_ID.txt