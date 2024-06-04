#!/bin/bash
#SBATCH --job-name=itrust-complement_naive_bayes
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --array=34,37,39,40,44,48
#SBATCH --time=00-01:00:00
#SBATCH --output=outputs/complement_naive_bayes-%A-%a.out
#SBATCH --error=errors/complement_naive_bayes-%A-%a.err

source ../.experiments_env/bin/activate

srun python complement_naive_bayes.py ${SLURM_ARRAY_TASK_ID} context2_ONLY-ACTION-SPREAD20_K3_H4_P12-BINARY
