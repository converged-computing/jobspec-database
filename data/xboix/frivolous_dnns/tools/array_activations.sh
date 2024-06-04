#!/bin/bash
#SBATCH -c1
#SBATCH --array=4
#SBATCH --job-name=get_activations
#SBATCH --mem=8GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 1:00:00
#SBATCH --chdir=/om/user/scasper/workspace/
#SBATCH --partition=cbmm

cd /om/user/scasper/workspace/
singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/scasper/redundancy/get_activations.py ${SLURM_ARRAY_TASK_ID}

# make sure to set the seed in imagenet_networks.py