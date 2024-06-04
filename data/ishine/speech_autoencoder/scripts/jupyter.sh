#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=bme_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --mem=32GB
#SBATCH --output=/hpc/data/home/bme/guochx/jupyter.log
module load 7/compiler/cuda/11.4
source /hpc/data/home/bme/guochx/.bashrc
conda activate torch18

# cat /etc/hosts
nvidia-smi
which python
srun jupyter notebook --no-browser --port=8888
# jupyter lab --ip=0.0.0.0 --port=8888