#!/bin/bash 
#SBATCH --partition=gpuq 
#SBATCH --qos=gpu 
#SBATCH --job-name=gpu_basics
#SBATCH --output=gpu_basics_french_8_training.%j.out 
#SBATCH --error=gpu_basics_french_8_training_error.%j.out 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=5
#SBATCH --gres=gpu:A100.40gb:1
#SBATCH --export=ALL
#SBATCH --time=0-11:00:00 

set echo 
umask 0022 

# to see ID and state of GPUs assigned
nvidia-smi

## Load the necessary modules
module load gnu10


export CUDA_VISIBLE_DEVICES=0


source /scratch/bpanigr/virtualenv-sled/bin/activate


## Execute you
python /scratch/bpanigr/nlp-final-project/SLED/examples/seq2seq/run.py /scratch/bpanigr/nlp-final-project/SLED/examples/seq2seq/configs/data/squad.json /scratch/bpanigr/nlp-final-project/SLED/examples/seq2seq/configs/model/bart_base_sled.json /scratch/bpanigr/nlp-final-project/SLED/examples/seq2seq/configs/training/base_training_args.json --output_dir /scratch/bpanigr/output_sled_fench_slurm --per_device_train_batch_size=8 --per_device_eval_batch_size=2

