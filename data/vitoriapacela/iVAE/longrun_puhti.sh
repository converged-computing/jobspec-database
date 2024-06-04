#!/bin/bash
#SBATCH --job-name=iVAE
#SBATCH --account=Project_2002842
#SBATCH --partition=longrun
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1-00:00:00

module purge
# module load gcc/8.3.0
module load pytorch/1.4

srun python3 main.py --config test-full-ivae-u.yaml --n-sims 1 --m 0.5 --s 0