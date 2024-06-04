#!/bin/zsh
#SBATCH -p mldlc_gpu-rtx2080 # alldlc_gpu-rtx2080
#SBATCH --gres=gpu:8
#SBATCH -J dino_neps_hpo
#SBATCH -t 5-23:59:59  # 23:59:59
#SBATCH --array 0-9999%10

#source /home/ferreira/.zshrc
source /home/ferreira/.profile
source activate dino

python -m main_dino --arch vit_small --data_path /data/datasets/ImageNet/imagenet-pytorch/train --output_dir /work/dlclarge2/ferreira-dino/metassl-dino/experiments/dino/$EXPERIMENT_NAME --batch_size_per_gpu 40 --is_neps_run --epochs 100 --world_size 8 --gpu 8
