#!/bin/bash
#SBATCH --nodes 1
#SBATCH -c 9
#SBATCH -p klab-gpu
#SBATCH --gres=gpu:H100.80gb:1
#SBATCH --error=ecoset_multi_error.o%j
#SBATCH --output=ecoset_multi_output.o%j

echo "running in shell: " "$SHELL"
## Please add any modules you want to load here, as an example we have commented out the modules
## that you may need such as cuda, cudnn, miniconda3, uncomment them if that is your use case 

spack load cuda@11.8.0
spack load cudnn@8.6.0.163-11.8
spack load miniconda3
eval "$(conda shell.bash hook)"
conda activate h100


python train.py