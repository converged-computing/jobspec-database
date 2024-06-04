#!/usr/bin/env bash
#SBATCH --job-name joncoron
#SBATCH --partition batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jkarls15@student.aau.dk
#SBATCH --time 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --chdir=/user/student.aau.dk/jkarls15

#srun singularity pull docker://nvcr.io/nvidia/tensorflow:20.02-tf1-py3

#srun singularity exec --nv tensorflow_20.02-tf1-py3.sif pip install --no-cache-dir -r requirements.txt
srun singularity exec --nv tensorflow_20.02-tf1-py3.sif python P10-ExoskeletonTransferLearning/meta_temp.py

# && are cursed characters