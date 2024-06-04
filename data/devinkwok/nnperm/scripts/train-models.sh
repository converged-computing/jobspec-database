#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --output=train-models-%j.out
#SBATCH --error=train-models-%j.err

# hparams
# MODEL=(cifar_vgg_16_8 cifar_vgg_16_16 cifar_vgg_16_32 cifar_vgg_16_128 cifar_vgg_16_256)
MODEL=(cifar_vgg_16_64 cifar_resnet_20_64)
# cifar10
DATASET=(svhn cifar100)
REPLICATE=($(seq 1 1 10))

source ./open_lth/slurm-setup.sh svhn cifar100
cd open_lth

parallel --delay=15 --linebuffer --jobs=3  \
    python open_lth.py lottery  \
        --default_hparams={2}  \
        --dataset_name={3}  \
        --replicate={1}  \
        --levels=0  \
        --training_steps=160ep  \
        --save_every_n_epochs=10  \
        --batchnorm_replace="layernorm"  \
        --pretrain  \
            --pretrain_dataset_name={3}  \
            --pretrain_warmup_steps="1ep"  \
            --pretrain_training_steps=10ep  \
            --pretrain_save_every_n_epochs=1  \
  ::: ${REPLICATE[@]}  \
  ::: ${MODEL[@]}  \
  ::: ${DATASET[@]}  \
