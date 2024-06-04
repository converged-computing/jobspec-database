#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=yongming_han@brown.edu
#SBATCH --job-name="apw_single"
#SBATCH --output="apw_single.%j.%N.out"
#SBATCH --error="apw_single.%j.err"
#SBATCH -n 1
#SBATCH --export=ALL
#SBATCH -c 1
#SBATCH -t 8:00:00 
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=batch

module load julia/1.5.0
cd /users/yh31/scratch/projects/gigaword_64k/
julia --project=@. /users/yh31/scratch/projects/gigaword_64k/src/apw_mapper_single.jl