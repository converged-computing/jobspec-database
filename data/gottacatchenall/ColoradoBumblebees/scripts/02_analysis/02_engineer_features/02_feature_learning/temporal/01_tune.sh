#!/bin/bash
#SBATCH --account=def-gonzalez
#SBATCH --mem=16G               # memory per node
#SBATCH --gpus-per-node=1
#SBATCH --time=1-00:00

module load cuda
module load julia/1.8.5
module load cudnn 

export JULIA_DEPOT_PATH="/project/def-gonzalez/mcatchen/JuliaEnvironments/COBees"
export CLUSTER="true"

julia ae_tuning.jl
