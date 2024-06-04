#!/bin/bash
#SBATCH --job-name=lion
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
##SBATCH --mail-type=BEGIN
##SBATCH --mail-user=jy3694@nyu.edu

cat<<EOF
Job starts at: $(date)

EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi
export DATA="lion"
export CLASS="lion"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="/scratch/jy3694/dataset/dreambooth/training/ours/$DATA"    # Path to instance images
export CLASS_DIR="/scratch/jy3694/dataset/dreambooth/regularization/ours/$CLASS"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
export OUTPUT_DIR="/scratch/jy3694/dreambooth_xl/runs/ours/$DATA"

singularity exec --nv --overlay $SCRATCH/venv_threestudio.ext3:ro \
      /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
      /bin/bash -c "source /ext3/env.sh; \
      accelerate launch train_dreambooth_lora_sdxl.py \
            --pretrained_model_name_or_path=$MODEL_NAME  \
            --instance_data_dir=$INSTANCE_DIR \
            --class_data_dir='$CLASS_DIR' \
            --pretrained_vae_model_name_or_path=$VAE_PATH \
            --output_dir=$OUTPUT_DIR \
            --with_prior_preservation \
            --prior_loss_weight=1.0 \
            --mixed_precision=\"fp16\" \
            --instance_prompt=\"a photo of sks '$CLASS'\" \
            --class_prompt=\"a photo of '$CLASS'\" \
            --resolution=1024 \
            --train_batch_size=1 \
            --gradient_accumulation_steps=4 \
            --learning_rate=1e-5 \
            --lr_scheduler=\"constant\" \
            --lr_warmup_steps=0 \
            --num_class_images=200 \
            --max_train_steps=800 \
            --validation_prompt=\"A photo of sks '$CLASS' in a bucket\" \
            --validation_epochs=25 \
            --seed=\"0\""