#!/bin/bash

#SBATCH --output=out_%A_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=23:59:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=certify_infer_minmax

PATCH_SIZE=$1
PATCH_STRIDE=$2
SIGMA=$3
RMODE=$4

module purge


singularity exec --nv \
	--overlay /scratch/aaj458/singularity_containers/my_pytorch.ext3:ro \
	/scratch/aaj458/singularity_containers/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
	/bin/bash -c "source /ext3/env.sh; python infer_certify_pretrained_salman.py imagenet -dpath /scratch/aaj458/data/datasets -mt resnet110 -mp salman_models/pretrained_models/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_$SIGMA/checkpoint.pth.tar -ni 10000 -ps $PATCH_SIZE -pstr $PATCH_STRIDE -sigma $SIGMA --N0 100 --N 10000 -o cifar10_SOTA/certify_results_salman_nopatch_3232/ --batch 400 -ns 32 --normalize"
