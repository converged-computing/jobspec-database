#!/bin/bash
#SBATCH --account=def-rsadve
#SBATCH --mail-user=<a.hebb@mail.utoronto.ca>
#SBATCH --mail-type=ALL

#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 # Remove this line to run using CPU only

#SBATCH --mem=24G
#SBATCH --time=02:34:56
#SBATCH --output=t-stdenv23-julia19-%N-%j.out

module purge
module load StdEnv/2023
module load cuda # Remove this line if not using a GPU

module load julia/1.9
module lead nvptx-tools

julia --project src/SkySurveillance.jl params-test.toml