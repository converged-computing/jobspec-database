#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/train_dreambooth_%j.out
#SBATCH --error=logs/train_dreambooth%j.err
#SBATCH --gres=gpu:a100:1

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi

export INSTANCE_DIR="/gpfs/home/lt2504/dreambooth/Dreambooth-Stable-Diffusion/training_images/Sam_Altman"
export OUTPUT_DIR="/gpfs/scratch/lt2504/diffusers_out/dreambooth-control-8000step-prior-preserve"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"

source /gpfs/scratch/lt2504/bert/bin/activate

accelerate launch train_dreambooth_control.py   --pretrained_model_name_or_path=$MODEL_NAME    --instance_data_dir=$INSTANCE_DIR --class_data_dir="class_dir"  --output_dir=$OUTPUT_DIR   --instance_prompt="a photo of sks" --class_prompt="photo of a man"   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=1   --learning_rate=5e-6   --lr_scheduler="constant"   --lr_warmup_steps=0   --max_train_steps=8000  --with_prior_preservation

