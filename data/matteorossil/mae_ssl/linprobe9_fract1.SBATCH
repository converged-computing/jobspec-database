#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=lp_1M_blur
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
    --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
    --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
	/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	/bin/bash -c "source /ext3/env.sh; python -u main_linprobe.py \
												--model vit_small_patch16 --cls_token \
												--finetune /vast/mr6744/pretrained_models2/1M_blur/run5/checkpoint-500.pth \
												--batch_size 128 \
												--epochs 100 \
												--num_workers 16 \
												--output_dir /vast/mr6744/linear_evaluation/1M_blur_ckpt_500/fract1 \
												--log_dir /vast/mr6744/linear_evaluation/1M_blur_ckpt_500/fract1 \
												--blr 0.1 \
												--weight_decay 0.0 \
												--nb_classes 1000 \
												--data_path /imagenet \
												--frac_retained 0.010147"
