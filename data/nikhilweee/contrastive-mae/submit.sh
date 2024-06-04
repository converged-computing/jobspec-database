#!/bin/bash
# Add -e to stop on errors
# Add -x to print before executing

#SBATCH --mem=64GB
#SBATCH --time=00-06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output="sbatch_logs/%A_%x.txt"
#SBATCH --job-name=pretrain_cont_norm_kq

# The following options will not be applied
# SBATCH --nodelist="gr*"
# SBATCH --gres=gpu:rtx8000:1


singularity exec \
    --nv --overlay /scratch/nv2099/images/overlay-50G-10M.ext3:ro \
    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh;
    conda activate vit_mae;
    cd /home/nv2099/projects/vit_mae/mae;
    python -u main_pretrain.py --loss_type cont --batch_size 128 \
        --output_dir runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID};
    # python -u main_finetune.py --finetune runs/pretrain_cifar/cont/checkpoint-49.pth \
    #     --output_dir runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID};
    "


# Pretrain Script
# python main_pretrain.py --batch_size 40 --loss_type cont
# Finetune Script
# python main_finetune.py --batch_size 40 --finetune runs/pretrain_cifar/cont/checkpoint-49.pth