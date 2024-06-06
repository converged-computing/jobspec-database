#!/usr/bin/bash
#SBATCH -J resnet
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A40:4
#SBATCH -t 14:00:00
#SBATCH --switches=1
#SBATCH -o log/%A/log.out
#SBATCH -e log/%A/err.out

export LOGLEVEL=INFO

mkdir -p log/$SLURM_JOB_ID
cp $2 log/$SLURM_JOB_ID/data_cfg.toml
cp $3 log/$SLURM_JOB_ID/train_cfg.toml

srun $1 \
    --standalone \
    --nproc_per_node=4 \
    --rdzv-backend=c10d \
    src/train.py --data-cfg log/$SLURM_JOB_ID/data_cfg.toml --train-cfg log/$SLURM_JOB_ID/train_cfg.toml

