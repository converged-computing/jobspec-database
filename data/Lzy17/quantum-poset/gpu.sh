#!/usr/bin/env bash

#SBATCH --job-name=posets
#SBATCH --account=use320
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=poset.o%j.%N





module purge
module list
printenv

python3 spectrum.py
