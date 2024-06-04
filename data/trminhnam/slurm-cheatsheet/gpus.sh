#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --job-name=gpu_check
#SBATCH --output=logs/gpu.%j.out
#SBATCH --error=errors/gpu.%j.err
#SBATCH --gres=gpu:8
#SBATCH --nodelist=

module load anaconda/3
conda env list
activate viglue

python -c 'import torch; print(torch.cuda.is_available())'

python -m bitsandbytes
nvidia-smi
