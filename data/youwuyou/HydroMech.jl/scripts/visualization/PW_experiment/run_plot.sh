#!/bin/bash

#SBATCH --job-name=plotting
#SBATCH --output=plot.out
#SBATCH --error=plot.err
#SBATCH --time=48:00:00

module load cuda-11.8.0-gcc-11.2.0-kh2t6kp

# Run your commands here
julia --project HighResolutionPlot.jl