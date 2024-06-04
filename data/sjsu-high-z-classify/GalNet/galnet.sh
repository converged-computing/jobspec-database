#!/bin/sh
#SBATCH --partition=gpus
#SBATCH --gres=gpu:2
#SBATCH --job-name=m2_valid
#SBATCH --output=m2_valid.out
#SBATCH --error=m2_valid.err
#SBATCH --time=10-00:00:00
#SBATCH --mail-user=james.casey-clyde@sjsu.edu
#SBATCH --mail-type=ALL

module purge
module load intel-python3
conda activate galnet
PYTHONHASHSEED=0 python galnet/galnet.py --train --model model-2
