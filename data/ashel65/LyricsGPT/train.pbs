#!/bin/bash -l
#PBS -N yourjob
#PBS -l select=1:ncpus=24:mem=120gb:ngpus=2:gpu_model=a100,walltime=12:00:00
#PBS -j oe
#PBS -o /home/alshelt/freestyle_rap_generator/job_summary.txt

module add anaconda3/2022.05-gcc/9.5.0
module add cuda/11.6.2-gcc/9.5.0

echo "Activating alex..."
conda init bash
conda activate alex

echo "Navigating to directory..."
cd /home/alshelt/freestyle_rap_generator

echo "Running training..."
python run_clm.py \
    --model_name_or_path EleutherAI/gpt-neox-20b \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm > output.txt 2>&1