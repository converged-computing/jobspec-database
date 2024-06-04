#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err



# STANFORDCARS "densenet121", "densenet161", "resnet34", "resnet152", "vgg16", "vgg19"
python code/models_prototype_pruning.py --dataset STANFORDCARS --base_architecture densenet121 --batchsize 16 --optimize_last_layer --num_workers 3 --gpu_id 0 --checkpoint
python code/models_prototype_pruning.py --dataset STANFORDCARS --base_architecture densenet161 --batchsize 16 --optimize_last_layer --num_workers 3 --gpu_id 0 --checkpoint
python code/models_prototype_pruning.py --dataset STANFORDCARS --base_architecture resnet34 --batchsize 16 --optimize_last_layer --num_workers 3 --gpu_id 0 --checkpoint
python code/models_prototype_pruning.py --dataset STANFORDCARS --base_architecture resnet152 --batchsize 16 --optimize_last_layer --num_workers 3 --gpu_id 0 --checkpoint
python code/models_prototype_pruning.py --dataset STANFORDCARS --base_architecture vgg16 --batchsize 16 --optimize_last_layer --num_workers 3 --gpu_id 0 --checkpoint
python code/models_prototype_pruning.py --dataset STANFORDCARS --base_architecture vgg19 --batchsize 16 --optimize_last_layer --num_workers 3 --gpu_id 0 --checkpoint

echo "Finished."
