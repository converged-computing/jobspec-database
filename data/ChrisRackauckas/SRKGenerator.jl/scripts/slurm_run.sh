#!/bin/bash
#SBATCH -A uci131
#SBATCH --job-name="SRK"
#SBATCH --output="output/jOpt.%j.%N.out"
#SBATCH --partition=gpu-shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=crackauc@uci.edu
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH -t 12:00:00
module load cuda/7.0
module load cmake
/home/crackauc/julia-3c9d75391c/bin/julia /home/crackauc/.julia/v0.5/SRKGenerator/test/runtests.jl 2496 $1 $2
