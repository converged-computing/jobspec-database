#!/bin/bash

#SBATCH --job-name=AR_sim
#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=10g
#SBATCH --array=1-200
#SBATCH -t 2-00:00:00
#SBATCH --output=./SLURMOUT/slurm_log_%A-%a.out

module add matlab
matlab -nodesktop -nosplash -singleCompThread -r CMH_ar1 -logfile ./logfiles/testingArray_$SLURM_ARRAY_TASK_ID.out

