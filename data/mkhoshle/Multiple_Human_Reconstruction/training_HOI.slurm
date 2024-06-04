#!/bin/bash

#SBATCH --job-name="train_HOI"
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --partition=lgns
#SBATCH --exclude=lgn3,lgn4
#SBATCH --time=48500 # Time in Minutes
#SBATCH --output="/z/home/mkhoshle/Human_object_transform/log_slurm/HOI_train-%j_%N.out"
#SBATCH --mail-user=mkhoshle@umich.edu
#SBATCH --mail-type=ALL


module load cuda/10.2
conda activate /z/home/mkhoshle/env/romp2

export TORCH_HOME=/z/home/mkhoshle/Human_object_transform/trained_models

export WANDB_API_KEY=79be3522200691f5f2e70b838fb23f90d4577836
export WANDB_CONFIG_DIR=/z/home/mkhoshle/Human_object_transform
export WANDB_DIR=/z/home/mkhoshle/Human_object_transform
export WANDB_CACHE_DIR=/z/home/mkhoshle/Human_object_transform


echo `nvidia-smi`
#Transfer data to the local node
rsync -avP /z/home/mkhoshle/dataset/ROMP_datasets/3DPW  $TMPDIR

TRAIN_CONFIGS='configs/v1.yml'
GPUS=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.GPUS)
DATASET=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.dataset)
TAB=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.tab)


#source activate handpose
cd /z/home/mkhoshle/Human_object_transform
#CUDA_VISIBLE_DEVICES=${GPUS} nohup python -u -m torch.distributed.launch --nproc_per_node=4 HumanObj/train.py --GPUS=${GPUS} --configs_yml=${TRAIN_CONFIGS} --distributed_training=1

python  HumanObj_videos_ResNet/train.py --GPUS=${GPUS} --configs_yml='configs/v1.yml' 




