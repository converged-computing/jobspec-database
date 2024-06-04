#!/bin/bash

#SBATCH --partition=shared
#SBATCH --job-name="JULIA"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=op
#SBATCH --time=00:01:00
#SBATCH --mem=2G
#SBATCH -o slurm.%N.%j.out 
#SBATCH -e slurm.%N.%j.err

cd $SLURM_SUBMIT_DIR

module load julia

julia example.jl > result
