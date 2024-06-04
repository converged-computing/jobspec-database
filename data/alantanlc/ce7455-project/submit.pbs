#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=96G:ngpus=1:ompthreads=16
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N t5-base_xs
#PBS -P Personal
cd ~
module load singularity/latest
cd winogrande
singularity exec ~/scratch/pyt_winogrande_new.simg python ./scripts/run_experiment_t5.py \
--model_type t5 \
--model_name_or_path "t5-base" \
--data_size "xs" \
--task_name winogrande_ps \
--do_eval \
--do_lower_case \
--data_dir ./data \
--max_seq_length 100 \
--per_gpu_eval_batch_size 8 \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 8 \
--learning_rate 1e-3 \
--num_train_epochs 15 \
--output_dir "./models/t5-base_xs" \
--do_train \
--logging_steps 1000 \
--save_steps 1000 \
--seed 42 \
--data_cache_dir ./data/cache/ \
--warmup_pct 0.1 \
--overwrite_output_dir \
--evaluate_during_training \
# --multi_task_perc 99999999 \
# --save_mem \
# --fp16 \
# --fp16_opt_level O0