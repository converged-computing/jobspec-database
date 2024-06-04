#!/bin/bash
#SBATCH -J pfenet
#SBATCH -N 1
#SBATCH -o results/pfenet.out
#SBATCH -e results/pfenet.err
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7000
#SBATCH --mail-type=START,END
#SBATCH --time=01:00:00

echo "Executing on $HOSTNAME"
date
module load nvidia/latest
module load cudnn/latest
CUDA_LAUNCH_BLOCKING=1 python3 test_pfenet.py
