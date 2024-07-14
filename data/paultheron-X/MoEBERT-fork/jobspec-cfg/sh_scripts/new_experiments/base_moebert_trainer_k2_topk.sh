#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
echo "Script name is: $0"
echo "Task name is $1"
echo "experiment name is $2"
echo "Distillation is $3"
echo "Weight decay is $4"

export LOCAL_OUTPUT="/home/gridsan/$(whoami)/MoEBERT-fork/results"

export output_dir="$LOCAL_OUTPUT/$1"
export saving_dir=$output_dir/"new_moebert_k2_experiment_$2" # Must correspond to the line in the excel hyperparameter tuning file

if [ $1 = 'cola' ]
then
    export original_model_dir="ModelTC/bert-base-uncased-cola"
    export metric_for_best_model="matthews_correlation"
    export batch_size=8
    export learning_rate=2e-5
    export eval_steps=200
    export num_epoch=15
elif [ $1 = 'rte' ]
then
    export original_model_dir="anirudh21/bert-base-uncased-finetuned-rte"
    export metric_for_best_model="accuracy"
    export batch_size=8
    export learning_rate=1e-5
    export eval_steps=200
    export num_epoch=15
elif [ $1 = 'mrpc' ]
then
    export original_model_dir='Intel/bert-base-uncased-mrpc'
    export metric_for_best_model="f1"
    export batch_size=8
    export learning_rate=3e-5
    export eval_steps=200
    export num_epoch=15
elif [ $1 = 'sst2' ]
then
    export original_model_dir='gchhablani/bert-base-cased-finetuned-sst2'
    export metric_for_best_model="accuracy"
    export batch_size=16
    export learning_rate=2e-5
    export eval_steps=1000
    export num_epoch=10
elif [ $1 = 'qqp' ]
then
    export original_model_dir='gchhablani/bert-base-cased-finetuned-qqp'
    export metric_for_best_model="f1"
    export batch_size=64
    export learning_rate=3e-5
    export eval_steps=2000
    export num_epoch=10
elif [ $1 = 'mnli' ]
then
    export original_model_dir='textattack/bert-base-uncased-MNLI'
    export metric_for_best_model="accuracy"
    export batch_size=64
    export learning_rate=5e-5
    export eval_steps=2000
    export num_epoch=10
elif [ $1 = 'qnli' ]
then
    export original_model_dir='textattack/bert-base-uncased-QNLI'
    export metric_for_best_model="accuracy"
    export batch_size=32
    export learning_rate=2e-5
    export eval_steps=2000
    export num_epoch=10
elif [ $1 = 'squad' ]
then
    export original_model_dir='csarron/bert-base-uncased-squad-v1'
    export metric_for_best_model="f1"
    export batch_size=32
    export learning_rate=3e-5
    export eval_steps=2000
    export num_epoch=10
elif [ $1 = 'squad_v2' ]
then
    export original_model_dir='deepset/bert-base-uncased-squad2'
    export metric_for_best_model="f1"
    export batch_size=16
    export learning_rate=3e-5
    export eval_steps=2000
    export num_epoch=10
fi

export LOCAL_SEED=2

echo "Number of epochs is $num_epoch"

if [ $1 = 'squad_v2' ] 
then
    python examples/question-answering/run_qa.py \
    --model_name_or_path $original_model_dir \
    --dataset_name $1 \
    --version_2_with_negative \
    --per_device_train_batch_size $batch_size \
    --weight_decay $4 \
    --learning_rate $learning_rate \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 384 \
    --num_train_epochs $num_epoch \
    --output_dir $saving_dir/model \
    --overwrite_output_dir \
    --logging_steps 60 \
    --logging_dir $saving_dir/log \
    --report_to tensorboard \
    --evaluation_strategy epoch \
    --eval_steps $eval_steps \
    --load_best_model_at_end False \
    --warmup_ratio 0.0 \
    --seed $LOCAL_SEED \
    --ignore_data_skip True \
    --fp16 \
    --moebert moe \
    --moebert_distill $3 \
    --moebert_expert_num 4 \
    --moebert_expert_dim 768 \
    --moebert_expert_dropout 0.1 \
    --moebert_load_balance 1.0 \
    --moebert_load_importance $LOCAL_OUTPUT/$1/experiment_${1}_template/importance_$1.pkl \
    --moebert_route_method topk \
    --moebert_share_importance 512 \
    --moebert_k 2 
elif [ $1 = 'squad' ] 
then
    python examples/question-answering/run_qa.py \
    --model_name_or_path $original_model_dir \
    --dataset_name $1 \
    --per_device_train_batch_size $batch_size \
    --weight_decay $4 \
    --learning_rate $learning_rate \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 384 \
    --num_train_epochs $num_epoch \
    --output_dir $saving_dir/model \
    --overwrite_output_dir \
    --logging_steps 60 \
    --logging_dir $saving_dir/log \
    --report_to tensorboard \
    --evaluation_strategy epoch \
    --eval_steps $eval_steps \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --warmup_ratio 0.0 \
    --seed $LOCAL_SEED \
    --ignore_data_skip True \
    --fp16 \
    --moebert moe \
    --moebert_distill $3 \
    --moebert_expert_num 4 \
    --moebert_expert_dim 768 \
    --moebert_expert_dropout 0.1 \
    --moebert_load_balance 1.0 \
    --moebert_load_importance $LOCAL_OUTPUT/$1/experiment_${1}_template/importance_$1.pkl \
    --moebert_route_method topk \
    --moebert_share_importance 512 \
    --moebert_k 2 
else 
    python examples/text-classification/run_glue.py \
    --model_name_or_path $original_model_dir \
    --task_name $1 \
    --per_device_train_batch_size $batch_size \
    --weight_decay $4 \
    --learning_rate $learning_rate \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --num_train_epochs $num_epoch \
    --output_dir $saving_dir/model \
    --overwrite_output_dir \
    --logging_steps 20 \
    --logging_dir $saving_dir/log \
    --report_to tensorboard \
    --evaluation_strategy steps \
    --eval_steps $eval_steps \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --metric_for_best_model $metric_for_best_model \
    --warmup_ratio 0.0 \
    --seed $LOCAL_SEED \
    --ignore_data_skip True \
    --fp16 \
    --moebert moe \
    --moebert_distill $3 \
    --moebert_expert_num 4 \
    --moebert_expert_dim 768 \
    --moebert_expert_dropout 0.1 \
    --moebert_load_balance 1.0 \
    --moebert_load_importance $LOCAL_OUTPUT/$1/experiment_${1}_template/importance_$1.pkl \
    --moebert_route_method topk \
    --moebert_share_importance 512 \
    --moebert_k 2 
fi