#!/bin/bash
#SBATCH --array=1-1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
# use case: sbatch run_prost.sh <domain> <online> <time>
source prost.sh $1 $SLURM_ARRAY_TASK_ID $3
