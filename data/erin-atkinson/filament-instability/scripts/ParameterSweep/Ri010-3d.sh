#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --job-name=filament-instability-Ri010-3d
#SBATCH --output=../scratch/logs/filament-instability/Ri010-3d.txt
module load cuda/11.0.3

cd ~/filament-instability

./../julia-1.8.5/bin/julia -t 8 --project="env" -- src/simulation.jl 3d parameters/ParameterSweep/Ri010.txt 1 25 100 ../scratch/filament-instability/Ri010-3d
