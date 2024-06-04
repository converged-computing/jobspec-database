#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=OSP
#SBATCH -n 1
#SBATCH -c 6
#SBATCH --mem=50000
#SBATCH -o out.txt
#SBATCH -e err.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load cuda/10.0.130
module load gnu7
module load openmpi3
module load anaconda/3.6
source activate /opt/ohpc/pub/apps/tensorflow_2.0.0
python3.7 -m pip install --user scikit-image
python3.7 -m pip install --user nibabel

# gpu:tesla-smx2:1
srun -n 1 python3 src/train.py
