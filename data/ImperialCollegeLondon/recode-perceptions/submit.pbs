#!/bin/sh
#PBS -lwalltime=24:00:00
#PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000                           # Requested HPC resources can be changed

module load anaconda3/personal
module load cuda/10.2
source activate perceptions

export PYTHONPATH=$HOME/user/.../recode-perceptions/                     			# Your path to recode-perceptions to run deep_cnn as module
export WB_KEY=KEY                                                                   # WANDB KEY
export WB_PROJECT="recode-perceptions"                                              # WANDB PROJECT NAME - if none will create uncategorized project folder
export WB_USER="username"                                                           # WANDB USER

python -m deep_cnn \
--epochs=50                             \
--batch_size=156                        \
--model='resnet18'                      \
--lr=1e-3                               \
--data_dir=input/places365_standard/                    \
--root_dir=$PYTHONPATH                  \
--wandb=True                            \
--run_name=resnet18_epochs50_lr1e-3_batch_156      \

conda deactivate
