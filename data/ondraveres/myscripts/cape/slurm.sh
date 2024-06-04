#!/bin/bash
#SBATCH --job-name=exp2
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --mem=40G
echo Hello
sleep 10
echo Starting

module load Julia
stdbuf -o0 -e0 julia  --color=no -O3 cape_explanations.jl  --task $1  #-i $SLURM_ARRAY_TASK_ID