#!/bin/bash
#SBATCH --job-name=train_spec_norm
#SBATCH --output=outLogs/train_model_%A_%a.out
#SBATCH --error=outLogs/train_model_%A_%a.err
#SBATCH --mem=100Gb
#SBATCH -N 1
#SBATCH --cpus-per-task=20
#SBATCH --time=1-00:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:a100:4
#SBATCH --array=0-11

source /etc/profile.d/modules.sh
module use /cm/shared/modulefiles
module add openmind/miniconda

# module load /openmind/miniconda

source activate /om/user/rphess/conda_envs/pytorch_2_tv

#module add openmind/cudnn/11.5-v8.3.3.40
#module add openmind/cuda/12.3

which python3
python3 train.py --job_id $SLURM_ARRAY_TASK_ID \
                 --gpus 4 --num_workers 5 \
