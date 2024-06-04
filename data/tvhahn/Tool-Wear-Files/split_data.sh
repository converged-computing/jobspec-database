#!/bin/bash
#SBATCH --time=00:10:00 # 10 minutes
#SBATCH --array=1-3
#SBATCH --mem=1G
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $
## How to use arrays: https://docs.computecanada.ca/wiki/Job_arrays

echo "Starting task $SLURM_ARRAY_TASK_ID"
DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" input_folders)

# Place the code to execute here
module load scipy-stack/2019b
python create_split_data.py $DIR
