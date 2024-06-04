#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00

singularity exec -e docker://brainlife/mcr:r2019a ./compiled/main config.json

