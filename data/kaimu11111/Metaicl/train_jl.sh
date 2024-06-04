#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/%j.out                              
#SBATCH --error=log/%j.out
#SBATCH --job-name=metaicl_train
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100-4

source activate metaicl
nvidia-smi

cd ../Metaicl
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mhong/li003755/.conda/envs/metaicl/lib/
# task=threeSimTask
# python train.py \
#   --task $task --k 16384 --test_k 16 --seed 100 --train_seed 1 --use_demonstrations --method channel --n_gpu 1 \
#   --batch_size 2 --lr 1e-5 --optimization 8bit-adam --out_dir checkpoints/channel-metaicl/$task \
#   --num_training_steps 10000

# task=threeDiffTask
# python train.py \
#   --task $task --k 16384 --test_k 16 --seed 100 --train_seed 1 --use_demonstrations --method channel --n_gpu 1 \
#   --batch_size 2 --lr 1e-5 --optimization 8bit-adam --out_dir checkpoints/channel-metaicl/$task \
#   --num_training_steps 10000 --num_samples 1038

task=oneSimTask
python train.py \
  --task $task --k 16384 --test_k 16 --seed 100 --train_seed 1 --use_demonstrations --method channel --n_gpu 1 \
  --batch_size 2 --lr 1e-5 --optimization 8bit-adam --out_dir checkpoints/channel-metaicl/$task \
  --num_training_steps 5000

task=oneDiffTask
python train.py \
  --task $task --k 16384 --test_k 16 --seed 100 --train_seed 1 --use_demonstrations --method channel --n_gpu 1 \
  --batch_size 2 --lr 1e-5 --optimization 8bit-adam --out_dir checkpoints/channel-metaicl/$task \
  --num_training_steps 5000 --num_samples 719

# --fp16
exit