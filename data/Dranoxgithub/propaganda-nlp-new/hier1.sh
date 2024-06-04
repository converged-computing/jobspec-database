#!/bin/bash
#SBATCH -p bhuwan --gres=gpu:1
#SBATCH --job-name=chunk
#SBATCH -o "./slurm_logs/%j.out"

RUN=$1
echo ${RUN}
echo $2
RUN_NAME="0.5_run_multiple_evals_new" # [8.678331090174966,1]
MODEL_NAME="hier1" # or "longformer"
OUTPUT_DIR="./runs/${RUN_NAME}"

hostname
nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv

# conda activate base
echo 'env started'

echo 'logging into wandb'
wandb login
export WANDB_PROJECT=debug
export WANDB_WATCH=all

echo 'invoking script to train chunk'
context_length=512
curr_split=${RUN}
time python3 run_chunk.py \
    --output_folder=${OUTPUT_DIR} \
    --model_name=${MODEL_NAME} \
    --model_num=$2 \
    --train_set="./split_data/processed_data/${curr_split}/train_${context_length}_chunk_CE.pkl" \
    --valid_set="./split_data/processed_data/${curr_split}/dev_${context_length}_chunk_CE.pkl" \
    --learning_rate=1e-5 \
    --random_seed=75 \
    --dropout=0.1 \
    --max_len=256 \
    --num_epochs=20 \
    --weight_decay=0.1 \
    --batch_size=8 \
    --num_labels=14 \
    --context_length=${context_length} \
    --aux_weight_train=0.5 \
    --aux_weight_eval=0.5 \
    --gold_file="split_data/datasets/${curr_split}/dev-task-flc-tc.labels.txt"
echo 'done'

    # --train_set="./processed_data/train_${context_length}_chunk.pkl" \
    # --valid_set="./processed_data/dev_${context_length}_chunk.pkl" \


# "./split_data/processed_data/${curr_split}/train_${context_length}_chunk_CE.pkl"