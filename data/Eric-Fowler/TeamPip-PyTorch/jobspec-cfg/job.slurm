#!/bin/bash
#SBATCH -p
#SBATCH gpu
#SBATCH --gpus=a100:1
#SBATCH --account=eee4773
#SBATCH --qos=eee4773
#SBATCH --mem-per-gpu=16gb
#SBATCH --time=00:30:00

module load python
module load cuda/11.1.0

# activate python environment
source env/bin/activate

nvidia-smi

python Script.py $TIME