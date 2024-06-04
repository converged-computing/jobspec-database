#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0:10:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_linear
#SBATCH --output=/scratch/mr6744/pytorch/outputs_slurm/%j.out

module purge

srun \
	singularity exec --nv \
	--overlay /scratch/mr6744/pytorch/overlay-25GB-500K.ext3:ro \
	/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	/bin/bash -c "source /ext3/env.sh; python -u eval_linear.py \
												--model 'vit_small_patch16' \
												--resume MODEL_PATH \
												--save_prefix INFORMATIVE_SAVE_PREFIX \
												--batch_size 128 \
												--epochs 50 \
												--num_workers 16 \
												--lr 0.0003 \
												--output_dir OUTPUT_DIR \
												--train_data_path TRAIN_DATA_PATH \
												--val_data_path VAL_DATA_PATH \
												--num_labels 1000"