#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-cpu=8G
#SBATCH -C gmem24
#SBATCH --job-name=DAC-1
#SBATCH --output=DAC-1.txt
#SBATCH --gres-flags=enforce-binding

echo "*"{,,,,,,,,,}
echo $SLURM_JOB_ID
echo "*"{,,,,,,,,,}

nvidia-smi
source ~/.bashrc
cd /home/sriniana/projects/mic/chest-pa1/DomainAdaptativeClassifier/xray_classification

CONDA_BASE=$(conda info --base) ;
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate mic
   
NUM_GPU=1
GPUS=0
PORT=12346

python3 train.py --config config.json