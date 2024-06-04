#!/bin/bash
#SBATCH -c1
#SBATCH --array=2-6
#SBATCH --job-name=demo
#SBATCH --mem=12GB
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -t 0-12:00:00
#SBATCH --chdir=/om/user/scasper/workspace/
#SBATCH --partition=normal

cd /om/user/scasper/workspace/

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/scasper/redundancy/train.py ${SLURM_ARRAY_TASK_ID}

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/scasper/redundancy/get_activations.py ${SLURM_ARRAY_TASK_ID}

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/scasper/redundancy/get_redundancy.py ${SLURM_ARRAY_TASK_ID}

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow1.14.simg \
python /om/user/scasper/redundancy/get_robustness.py ${SLURM_ARRAY_TASK_ID}
