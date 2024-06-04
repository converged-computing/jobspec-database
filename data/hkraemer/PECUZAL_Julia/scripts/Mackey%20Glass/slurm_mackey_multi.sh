#!/bin/bash

#SBATCH --qos=medium
#SBATCH --partition=standard
#SBATCH --job-name=mackey
#SBATCH --account=synet
#SBATCH --output=name-%j.out
#SBATCH --error=name-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8


module load julia/1.5.3
module load hpc
julia comm_mackey_ensemble.jl
