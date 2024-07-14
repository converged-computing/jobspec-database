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
    echo "No output dir passed"
    export LOCAL_OUTPUT="/home/gridsan/$(whoami)/MoEBERT-fork/results"
else
    echo "Given Output dir is ${13}"
    export LOCAL_OUTPUT=${13}
fi

export output_dir="$LOCAL_OUTPUT/$1"
export saving_dir=$output_dir/"moebert_k2_experiment_$2" # Must correspond to the line in the excel hyperparameter tuning file
export original_model_dir=$output_dir/"experiment_$1_finetuned_model"

if [ $1 = 'cola' ]
then
    export metric_for_best_model="matthews_correlation"
elif [ $1 = 'rte' ]
then
    export metric_for_best_model="accuracy"
elif [ $1 = 'mrpc' ]
then
    export metric_for_best_model="f1"
elif [ $1 = 'sst2' ]
then
    export metric_for_best_model="accuracy"
elif [ $1 = 'qqp' ]
then
    export metric_for_best_model="f1"
elif [ $1 = 'mnli' ]
then
    export metric_for_best_model="accuracy"
elif [ $1 = 'qnli' ]
then
    export metric_for_best_model="accuracy"
fi

if [ ${11} = 'True' ]
then
    if [ $1 = 'sst2' ]
    then
        export num_epochs=25
    else
        export num_epochs=15
    fi
else
    export num_epochs=10
fi

# check if the parameter 12 is passed or not
if [ -z "${12}" ]
then
    echo "No seed passed"
    export LOCAL_SEED=0
else
    echo "Seed passed is ${12}"
    export LOCAL_SEED=${12}
fi

echo "Number of epochs is $num_epochs"

if [ $1 = 'cola' ] || [ $1 = 'rte' ] || [ $1 = 'mrpc' ]
then
    python examples/text-classification/run_glue.py \
    --model_name_or_path $original_model_dir/model \
    --task_name $1 \
    --per_device_train_batch_size $3 \
    --weight_decay $4 \
    --learning_rate $5 \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --num_train_epochs 50 \
    --output_dir $saving_dir/model \
    --overwrite_output_dir \
    --logging_steps 20 \
    --logging_dir $saving_dir/log \
    --report_to tensorboard \
    --evaluation_strategy epoch \
    --load_best_model_at_end False \
    --metric_for_best_model $metric_for_best_model \
    --warmup_ratio 0.0 \
    --seed $LOCAL_SEED \
    --ignore_data_skip True \
    --fp16 \
    --moebert moe \
    --moebert_distill $8 \
    --moebert_expert_num 4 \
    --moebert_expert_dim 768 \
    --moebert_expert_dropout 0.1 \
    --moebert_load_balance 0.0 \
    --moebert_load_importance $original_model_dir/importance_$1.pkl \
    --moebert_route_method soft-tree \
    --moebert_share_importance 512 \
    --moebert_gate_entropy $6 \
    --moebert_gate_gamma $7 \
    --moebert_k 2
else
    python -m torch.distributed.launch --nproc_per_node=$num_gpus
    examples/text-classification/run_glue.py \
    --model_name_or_path $original_model_dir/model \
    --task_name $1 \
    --per_device_train_batch_size $3 \
    --weight_decay $4 \
    --learning_rate $5 \
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
    --eval_steps $9 \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --metric_for_best_model $metric_for_best_model \
    --warmup_ratio 0.0 \
    --seed $LOCAL_SEED \
    --ignore_data_skip True \
    --fp16 \
    --moebert moe \
    --moebert_distill $8 \
    --moebert_expert_num 4 \
    --moebert_expert_dim 768 \
    --moebert_expert_dropout 0.1 \
    --moebert_load_balance 0.0 \
    --moebert_load_importance $original_model_dir/importance_$1.pkl \
    --moebert_route_method soft-tree \
    --moebert_share_importance 512 \
    --moebert_gate_entropy $6 \
    --moebert_gate_gamma $7 \
    --moebert_k 2
fi