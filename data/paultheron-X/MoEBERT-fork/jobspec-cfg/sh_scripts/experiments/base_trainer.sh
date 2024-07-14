#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
echo "Script name is: $0"
echo "Task name is $1"
echo "experiment name is $2"
echo "Batch size is $3"
echo "weight decay is $4"
echo "learning rate is $5"
echo "eval steps is $6"
echo "Given output dir is $7"
export output_dir="$7/$1"
export saving_dir=$output_dir/"experiment_$2" # Must correspond to the line in the excel hyperparameter tuning file

#python -m torch.distributed.launch --nproc_per_node=$num_gpus \

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

if [ $1 = 'cola' ] || [ $1 = 'rte' ] || [ $1 = 'mrpc' ]
then
    python examples/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name $1 \
    --per_device_train_batch_size $3 \
    --weight_decay $4 \
    --learning_rate $5 \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --num_train_epochs 10 \
    --output_dir $saving_dir/model \
    --overwrite_output_dir \
    --logging_steps 20 \
    --logging_dir $saving_dir/log \
    --report_to tensorboard \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model $metric_for_best_model \
    --warmup_ratio 0.0 \
    --seed 0 \
    --fp16 
else
    python examples/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name $1 \
    --per_device_train_batch_size $3 \
    --weight_decay $4 \
    --learning_rate $5 \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --num_train_epochs 10 \
    --output_dir $saving_dir/model \
    --overwrite_output_dir \
    --logging_steps 20 \
    --logging_dir $saving_dir/log \
    --report_to tensorboard \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model $metric_for_best_model \
    --warmup_ratio 0.0 \
    --seed 0 \
    --fp16 
fi