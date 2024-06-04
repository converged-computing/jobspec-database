#!/bin/bash
#SBATCH --job-name=self_5_to_17_imps
#SBATCH --partition=reserved
#SBATCH --account=rac-2018-hpcg1742
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavithran.sridhar@gmail.com
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096
# #SBATCH --array=0-5
./benchmarking input.txt
# Use this with an array job to run multiple instances
# echo "Running task" $SLURM_ARRAY_TASK_ID
# ./instances.sh $SLURM_ARRAY_TASK_ID
