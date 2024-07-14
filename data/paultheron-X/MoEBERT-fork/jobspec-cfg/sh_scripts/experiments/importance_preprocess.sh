#!/bin/bash

export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
echo "Script name is: $0"
echo "Task name is $1"
echo "Given output dir is $2"
export output_dir="$2/$1"
export original_model_dir=$output_dir/"experiment_$1_finetuned_model/model"
export saving_dir=$output_dir/"moebert_experiment_$1" # Must correspond to the line in the excel hyperparameter tuning file

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
    python -m torch.distributed.launch --nproc_per_node=$num_gpus \
    examples/text-classification/run_glue.py \
    --model_name_or_path $original_model_dir \
    --task_name $1 \
    --preprocess_importance \
    --do_eval \
    --max_seq_length 128 \
    --output_dir $saving_dir/model \
    --overwrite_output_dir \
    --logging_steps 20 \
    --logging_dir $saving_dir/log \
    --report_to tensorboard \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end False \
    --metric_for_best_model $metric_for_best_model \
    --warmup_ratio 0.0 \
    --seed 0 \
    --weight_decay 0.0 \
    --fp16 
else
    python -m torch.distributed.launch --nproc_per_node=$num_gpus \
    examples/text-classification/run_glue.py \
    --model_name_or_path $original_model_dir \
    --task_name $1 \
    --preprocess_importance \
    --do_eval \
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
    --load_best_model_at_end False \
    --metric_for_best_model $metric_for_best_model \
    --warmup_ratio 0.0 \
    --seed 0 \
    --weight_decay 0.0 \
    --fp16 
fi