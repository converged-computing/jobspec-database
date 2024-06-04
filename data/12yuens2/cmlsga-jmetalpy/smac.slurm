#!/bin/bash

# Set resource requirements

#SBATCH --array=1-2
#SBATCH --mem=4000
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=24:05:00

# Logging

#SBATCH --output=smac-run.out
#SBATCH --error=smac-run.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sy6u19@soton.ac.uk

# load environment
module load python/3.7.3
source env/bin/activate

cd $HOME/cmlsga-jmetalpy
python src/tuning.py nsgaii ZDT${SLURM_ARRAY_TASK_ID} ga
