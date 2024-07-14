#!/bin/bash

export num_gpus=3
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
echo "Script name is: $0"



export LOCAL_OUTPUT="/home/gridsan/$(whoami)/MoEBERT-fork/results"

export output_dir="$LOCAL_OUTPUT/qnli"
export saving_dir=$output_dir/"qnli_hash_experiment" # Must correspond to the line in the excel hyperparameter tuning file
export original_model_dir=$output_dir/"experiment_qnli_finetuned_model"

export metric_for_best_model="accuracy"
export num_epochs=15


echo "Number of epochs is $num_epochs"

python examples/text-classification/run_glue.py \
    --model_name_or_path $original_model_dir/model \
    --task_name qnli \
    --per_device_train_batch_size 16 \
    --weight_decay 0.0 \
    --learning_rate 2e-5 \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --num_train_epochs $num_epochs \
    --output_dir $saving_dir/model \
    --overwrite_output_dir \
    --logging_steps 20 \
    --logging_dir $saving_dir/log \
    --report_to tensorboard \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --metric_for_best_model $metric_for_best_model \
    --warmup_ratio 0.0 \
    --seed 0 \
    --ignore_data_skip True \
    --fp16 \
    --moebert moe \
    --moebert_distill 2.0 \
    --moebert_expert_num 4 \
    --moebert_expert_dim 768 \
    --moebert_expert_dropout 0.1 \
    --moebert_load_balance 0.0 \
    --moebert_load_importance $original_model_dir/importance_qnli.pkl \
    --moebert_route_method hash-random \
    --moebert_share_importance 512 \
    --moebert_gate_entropy 1 \
    --moebert_gate_gamma 1