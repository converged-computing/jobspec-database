#!/bin/bash
#PBS -l walltime=72:00:00,select=1:ncpus=12:ompthreads=12:ngpus=1:mem=32gb
#PBS -m abe
#PBS -N jax_privacy
#PBS -M min01@student.ubc.ca
#PBS -o output_jax_privacy.txt
#PBS -e error_jax_privacy.txt
#PBS -A st-mijungp-1-gpu

# Change directory into the job dir
cd /arc/project/st-mijungp-1/yingchen/jax_privacy
# Set Python path
export PYTHONPATH=$PYTHONPATH:/arc/project/st-mijungp-1/yingchen/jax_privacy
# Load conda environment
source ~/.bashrc
# Activate conda environment
conda activate jax_privacy
# export CUDA_VISIBLE_DEVICES=""


python3 /arc/project/st-mijungp-1/yingchen/jax_privacy/experiments/image_classification/run_experiment.py --config=/arc/project/st-mijungp-1/yingchen/jax_privacy/experiments/image_classification/configs/cifar100_wrn_28_10_eps1_finetune.py --jaxline_mode=train_eval_multithreaded
