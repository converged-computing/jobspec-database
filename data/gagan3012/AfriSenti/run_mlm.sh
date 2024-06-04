#!/bin/bash
#SBATCH --time=10:59:59
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --account=rrg-mageed
#SBATCH --job-name=train_image_v2
#SBATCH --output=/lustre07/scratch/gagan30/arocr/logs/%x.out
#SBATCH --error=/lustre07/scratch/gagan30/arocr/logs/%x.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

module load python/3.8 scipy-stack gcc arrow cuda cudnn httpproxy 

source ~/ENV38_default/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python run_mlm.py \
    --model_name_or_path ../../../models/afro-xlmr-base \
    --train_file ../Pretrain/train.csv \
    --validation_file ../Pretrain/test.csv \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --do_train \
    --do_eval \
    --output_dir ../../../models/afro-xlmr-base-lm \
    --num_train_epochs 25 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing True \
    --save_total_limit 2 \
    --max_seq_length 128 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_steps 100 \
    --seed 42 \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --greater_is_better True \
    --line_by_line \
    --fp16 \
