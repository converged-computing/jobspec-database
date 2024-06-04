#!/usr/bin/env bash

#SBATCH --account=COMS030144
#SBATCH --job-name=cw

#SBATCH --partition=teach_gpu
#SBATCH --nodes=1
# Specifying the output and error file paths
#SBATCH -o ./bc4_out/log_%j.out # STDOUT out
#SBATCH -e ./bc4_out/log_%j.err # STDERR out
#SBATCH --gres=gpu:2
#SBATCH --time=3:00:00
#SBATCH --mem=16GB

# Create the output directory if it doesn't exist
mkdir -p ./bc4_out

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"

# Run the python script
python3 main.py --epochs 10 --learning-rate 0.2 --sgd-momentum 0.93 --op sgd --model ChunkResCNN1 --conv-length 256 --conv-stride 256