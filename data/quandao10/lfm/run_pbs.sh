#!/bin/bash
#Below is the queue type, use ai for GPU usages
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=32:mem=128G:ngpus=1
#PBS -l walltime=48:00:00
#PBS -P Personal
#PBS -N flow-matching-redressing-cifar10
#PBS -m abe
#PBS -M tuanbinhs@gmail.com

source $HOME/miniconda3/bin/activate ottest
cd $HOME/scratch/work/lfm

set -x
set -e

	       
CUDA_VISIBLE_DEVICES=0 python train_redressing_flow.py \
		       --exp experiment_cifar_redressing \
		       --dataset cifar10 \
		       --minibatch_ot 0 \
		       --batch_size 128 \
		       --num_epoch 600 \
		       --image_size 32 \
		       --num_channels 256 \
		       --attn_resolution (8,) \
		       --num_res_blocks 2 \
		       --lr 1e-4 \
		       --num_process_per_node 1 \
		       --save_content \
		       --save_content_every 10 \

# CUDA_VISIBLE_DEVICES=0 python train_flow.py \
#                --exp experiment_cifar_default \
#                --dataset cifar10 \
# #               --datadir data \
#                --batch_size 128 \
#                --num_epoch 10 \
#                --image_size 32 \
# #               --num_in_channels 3 \
# #               --num_out_channels 3 \
#                --num_channels 256 \
# #               --ch_mult 1 2 2 2 \
#                --attn_resolution (8,) \
#                --num_res_blocks 2 \
#                --lr 1e-4 \
# 	       --num_process_per_node 1
# #               --label_dropout 0. \
#                --save_content \
#                --save_content_every 10 \


# CUDA_VISIBLE_DEVICES=0 python train_flow.py \
#                --exp experiment_cifar_minibatchOT \
#                --dataset cifar10 \
# 	       --minibatch_ot 1 \
# #               --datadir data \
#                --batch_size 128 \
#                --num_epoch 600 \
#                --image_size 32 \
# #               --num_in_channels 3 \
# #               --num_out_channels 3 \
#                --num_channels 256 \
# #               --ch_mult 1 2 2 2 \
#                --attn_resolution (8,) \
#                --num_res_blocks 2 \
#                --lr 1e-4 \
# 	       --num_process_per_node 1
# #               --label_dropout 0. \
#                --save_content \
#                --save_content_every 10 \
