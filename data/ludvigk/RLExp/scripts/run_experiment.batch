#!/bin/bash
#SBATCH --job-name=NDQN
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=job_output.out
#SBATCH --tasks-per-node=12
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=ludvig.killingberg@ntnu.no
#SBATCH --account=share-ie-idi
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --constraint="V100|A100"
#SBATCH --partition=GPUQ

module load Python/3.8.6-GCCcore-10.2.0

julia --optimize=3 --project=. scripts/run_experiment.jl
