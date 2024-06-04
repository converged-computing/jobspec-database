#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=00:20:00
#SBATCH --mem=8GB
#SBATCH --array=1-3
#SBATCH --err="hpc/logs/batch_gi_%a.err"
#SBATCH --output="hpc/logs/batch_gi_%a.out"
#SBATCH --job-name="make_gioj"

# Setup Python Environment
module load Singularity
module load CUDA/10.2.89

# Start singularity instance
singularity run main.simg -p gnnproject/analysis/run_batch_make_gioj.py -a $SLURM_ARRAY_TASK_ID