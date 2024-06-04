#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --gres=gpu:titanx:1
#SBATCH --array=1-30

####################################
# Declare variables
out_folder="results/cifar10/resnet20/bootstr"
model_type="ResNet20v1" # ResNet20v1, ResNet110v1, WideResNet28-10
use_case="cifar10" # cifar10, cifar100
initial_lr=0.1
l2_reg=0.0003
lr_schedule="sse" # sse, garipov, cifar (for ResNet20 only)
epochs=300
checkpoint_every=60
options="--bootstrapping" # "" or "--bootstrapping"

####################################


export CUDNN_PATH=$HOME/.conda/envs/TF_KERAS_3_GPU/lib/python3.10/site-packages/nvidia/cudnn
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$HOME/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
export TF_ENABLE_ONEDNN_OPTS=0

# If SLURM_ARRAY_TASK_ID is not set, set it to 1
if [ -z ${SLURM_ARRAY_TASK_ID+x} ]; then
    SLURM_ARRAY_TASK_ID=1
fi

# Run experiment
printf "\n\n* * * Run SGD for ID = $SLURM_ARRAY_TASK_ID. * * *\n\n\n"

python -m training \
    --id=$(printf "%02d" $SLURM_ARRAY_TASK_ID) \
    --seed=$SLURM_ARRAY_TASK_ID \
    --out_folder=$out_folder \
    --validation_split=0.0 \
    --model_type=$model_type \
    --data_augmentation \
    --nesterov \
    --optimizer="sgd" \
    --use_case=$use_case \
    --initial_lr=$initial_lr \
    --l2_reg=$l2_reg \
    --lr_schedule=$lr_schedule \
    --checkpointing \
    --checkpoint_every=$checkpoint_every \
    --epochs=$epochs \
    $options
