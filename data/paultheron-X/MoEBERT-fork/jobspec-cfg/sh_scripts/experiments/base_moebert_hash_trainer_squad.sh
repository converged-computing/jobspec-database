#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
echo "Script name is: $0"
echo "Task name is $1"
echo "experiment name is $2"
echo "Batch size is $3"
echo "weight decay is $4"
echo "learning rate is $5"
echo "entropy is $6"
echo "gamma is $7"
echo "distillation penalty is $8"
echo "eval steps is $9"
echo "1st stage best epoch is ${10}"
echo "Mode large is ${11}"
echo "Seed is ${12}"
echo "Given Output dir is ${13}"

# check if the parameter 13 for output dir is passed or not
if [ -z "${13}" ]
then
    export LOCAL_OUTPUT="/home/gridsan/$(whoami)/MoEBERT-fork/results"
    echo "No output dir passed, defaulting to $LOCAL_OUTPUT"
else
    echo "Given Output dir is ${13}"
    export LOCAL_OUTPUT=${13}
fi

export output_dir="$LOCAL_OUTPUT/squad_v2"
export saving_dir=$output_dir/"hash_routing_experiment_$2" # Must correspond to the line in the excel hyperparameter tuning file
export original_model_dir=$output_dir/"experiment_squad_v2_template"
export best_model_dir='deepset/bert-base-uncased-squad2'


export metric_for_best_model="f1"

export num_epochs=6

# check if the parameter 12 is passed or not
if [ -z "${12}" ]
then
    export LOCAL_SEED=0
    echo "No seed passed, defaulting to $LOCAL_SEED"
else
    echo "Seed passed is ${12}"
    export LOCAL_SEED=${12}
fi

echo "Number of epochs is $num_epochs"

python examples/question-answering/run_qa.py \
    --model_name_or_path $best_model_dir \
    --dataset_name squad_v2 \
    --version_2_with_negative \
    --per_device_train_batch_size 16 \
    --weight_decay 0.1 \
    --learning_rate 0.00003 \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 384 \
    --num_train_epochs $num_epochs \
    --output_dir $saving_dir/model \
    --overwrite_output_dir \
    --logging_steps 60 \
    --logging_dir $saving_dir/log \
    --report_to tensorboard \
    --eval_steps 2000 \
    --evaluation_strategy epoch \
    --load_best_model_at_end False \
    --warmup_ratio 0.0 \
    --seed $LOCAL_SEED \
    --ignore_data_skip True \
    --moebert moe \
    --moebert_distill 1.0 \
    --moebert_expert_num 4 \
    --moebert_expert_dim 768 \
    --moebert_expert_dropout 0.1 \
    --moebert_load_balance 0.1 \
    --moebert_load_importance $original_model_dir/importance_squad_v2.pkl \
    --moebert_route_method hash-random \
    --moebert_share_importance 512 \
    --moebert_gate_entropy 1 \
    --moebert_gate_gamma 1
