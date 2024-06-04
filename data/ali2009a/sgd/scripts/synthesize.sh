#!/bin/bash
#SBATCH --job-name=synthesize
#SBATCH --output=/home/aliarab/scratch/sgd/sim_data/data_params_m/output.log
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --account=def-ester

module load matlab/2018a
project_root="/home/aliarab/src/sgd/scripts"

cd $project_root
matlab -nodesktop -r synthesize_wrapper
