#!/bin/bash
#SBATCH -n 4
#SBATCH --array=0-999
#SBATCH --job-name=autoencoder
#SBATCH --mem=10GB
#SBATCH -t 04:30:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --workdir=/om/user/nprasad/aesap/subs/arch/

singularity exec -B /om:/om --nv /om/user/nprasad/singularity/tensorflow-1.8.0-gpu-py3.img \
python /om/user/nprasad/aesap/main.py /om/user/nprasad/aesap/ ${SLURM_ARRAY_TASK_ID} 0