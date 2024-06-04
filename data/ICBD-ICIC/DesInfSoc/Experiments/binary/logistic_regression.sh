#!/bin/bash
#SBATCH --job-name=itrust-logistic_regression
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=34,37,39,40,44,48
#SBATCH --time=00-01:00:00
#SBATCH --output=outputs/logistic_regression-%A-%a.out
#SBATCH --error=errors/logistic_regression-%A-%a.err

source ../.experiments_env/bin/activate

srun python logistic_regression.py ${SLURM_ARRAY_TASK_ID} context2_ONLY-ACTION-SPREAD20_K3_H4_P12-BINARY
