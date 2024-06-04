#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=200K_blur
#SBATCH --output=/scratch/mr6744/pytorch/outputs_slurm/%j.out

export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
echo "MASTER_ADDR="$MASTER_ADDR

module purge

srun \
	singularity exec --nv \
	--overlay /scratch/mr6744/pytorch/overlay-25GB-500K.ext3:ro \
	/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	/bin/bash -c "source /ext3/env.sh; python -u train_mae_nowds.py \
												--model 'mae_vit_small_patch16' \
												--batch_size_per_gpu 128 \
												--num_workers 16 \
												--lr 0.0003 \
												--min_lr 0.0003 \
												--weight_decay 0.0 \
												--data_path /vast/mr6744/SAYCAM_blur/ \
												--output_dir /vast/mr6744/pretrained_models/run3 \
												--save_prefix vit_s_16_200K_blur \
												--resume /vast/mr6744/pretrained_models/run2/vit_s_16_200K_blur_checkpoint.pth"