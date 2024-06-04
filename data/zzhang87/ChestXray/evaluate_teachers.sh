#!/bin/bash
#
# This script performs the following operations:
# 1. Fine-tunes a MobileNet model on the given training set.
# 2. Evaluates the model on the validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_resnet_v1_50_on_flowers.sh

#-----------------SLURM Parameters-------------------------------
#SBATCH -p terramepp
#SBATCH -n 8
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -D /home/n-z/zzhang52/Insight/ChestXray/
#SBATCH -J resnet_50_small_train
#SBATCH -A ChestXray
#PBS -M zzhang52@illinois.edu
#PBS -m abe

#-----------------Load Modules------------------------------------
# module load Tensorflow-GPU/1.2.1-IGB-gcc-4.9.4-Python-3.6.1

#-----------------Define Variables-------------------------------
MODEL_DIR=/home/paperspace/Insight

# Where the dataset is saved to.
DATA_DIR=/home/paperspace/Insight/CXR/5_part_all

IMAGE_DIR=/home/paperspace/Insight/CXR/images/

PARTITION_NUM=5

BATCH_SIZE=128


#-----------------------Execute Commands------------------------------

for teacher in `seq 1 $PARTITION_NUM`
do
python evaluate_model.py \
  --data_dir=${DATA_DIR} \
  --image_dir=${IMAGE_DIR} \
  --model_dir=${MODEL_DIR}/teacher_${teacher} \
  --partition_id=${teacher} \
  --partition_num=${PARTITION_NUM} \
  --batch_size=${BATCH_SIZE} \
  --split_name='val'

done


for teacher in `seq 1 $PARTITION_NUM`
do
python evaluate_model.py \
  --data_dir=${DATA_DIR} \
  --image_dir=${IMAGE_DIR} \
  --model_dir=${MODEL_DIR}/teacher_${teacher} \
  --batch_size=${BATCH_SIZE} \
  --split_name='test'

done
