#!/bin/sh
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 47:59:00
#SBATCH --job-name="manet-run"
#SBATCH --gpus=v100-32:1
#SBATCH --output=/ocean/projects/iri180005p/mmehta1/ControlNet/logs/manet.out

set -x
cd /ocean/projects/iri180005p/mmehta1/ControlNet
pwd
conda deactivate
module load anaconda3/2020.11
conda activate control
nvidia-smi
python train_manet.py