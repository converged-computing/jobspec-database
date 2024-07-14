#!/bin/bash

export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="/home/gridsan/ptheron/MoEBERT-fork/results/"
echo "Script name is: $0"
echo "Task name is $1"
echo "experiment name is $2"
echo "Batch size is $3"
echo "weight decay is $4"
echo "learning rate is $5"
echo "eval steps is $6"
export saving_dir=$output_dir/"squad_experiment_$2" # Must correspond to the line in the excel hyperparameter tuning file

export metric_for_best_model="f1"

if [ $1 = "squad" ]; then
    python examples/question-answering/run_qa.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name squad \
    --per_device_train_batch_size $3 \
    --weight_decay $4 \
    --learning_rate $5 \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 10 \
    --output_dir $saving_dir/model \
    --overwrite_output_dir \
    --logging_steps 20 \
    --logging_dir $saving_dir/log \
    --report_to tensorboard \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --warmup_ratio 0.0 \
    --seed 0 \
    --fp16 

elif [ $1 = "squad_v2" ]; then
    python examples/question-answering/run_qa.py \
    --model_name_or_path bert-base-uncased \
    --dataset_name squad_v2 \
    --version_2_with_negative \
    --per_device_train_batch_size $3 \
    --weight_decay $4 \
    --learning_rate $5 \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 15 \
    --output_dir $saving_dir/model \
    --overwrite_output_dir \
    --logging_steps 20 \
    --logging_dir $saving_dir/log \
    --report_to tensorboard \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --warmup_ratio 0.0 \
    --seed 0 \
    --fp16
else
    echo "Task name not recognized"
fi