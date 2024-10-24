#!/bin/bash

#SBATCH --job-name="Train Tabular (ECCCo)"
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=general
#SBATCH --mem-per-cpu=4GB
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module use /opt/insy/modulefiles          # Use DAIC INSY software collection
module load openmpi

srun julia --project=experiments experiments/run_experiments.jl -- data=gmsc,german_credit,california_housing output_path=results only_models > experiments/train_tabular.log
