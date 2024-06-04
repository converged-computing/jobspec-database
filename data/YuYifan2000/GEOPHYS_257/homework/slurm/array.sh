#!/bin/sh
#SBATCH --array=1-100
#SBATCH --ntasks=1
#SBATCH --partition preempt
#SBATCH --time=0-2
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --requeue

spack load python@3.10.8
python ./pi_random.py $SLURM_ARRAY_TASK_ID
