#!/bin/bash
# FILENAME: job.sh
#SBATCH --output=myjob.out
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --job-name cifar-resnet
#SBATCH --output ./out.out
#SBATCH --error ./err.err
#SBATCH --job-name testing_models

module purge
module load anaconda
module load use.own
source activate d22env
conda info --envs
echo -e "module loaded"


python -u cifar_resnet18.py
