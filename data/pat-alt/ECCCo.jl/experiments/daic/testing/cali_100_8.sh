#!/bin/bash

#SBATCH --job-name="Grid-search California Housing (ECCCo)"
#SBATCH --time=00:35:00
#SBATCH --ntasks=14
#SBATCH --cpus-per-task=14
#SBATCH --partition=general
#SBATCH --mem-per-cpu=2GB
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 

module use /opt/insy/modulefiles          # Use DAIC INSY software collection
module load openmpi

source experiments/slurm_header.sh

srun julia --project=experiments --threads $SLURM_CPUS_PER_TASK experiments/run_experiments.jl -- data=california_housing output_path=results mpi grid_search n_individuals=10 threaded > experiments/grid_search_california_housing.log

