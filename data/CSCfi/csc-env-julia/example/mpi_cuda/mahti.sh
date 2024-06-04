#!/bin/bash
#SBATCH --job-name=cuda
#SBATCH --account=project_2001659
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:1
module load julia/1.8.5
srun julia --project=. test.jl
