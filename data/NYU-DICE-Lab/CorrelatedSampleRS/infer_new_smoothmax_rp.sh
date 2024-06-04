#!/bin/bash

#SBATCH --output=out_%A_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=47:59:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=certify_infer_minmax

PATCH_SIZE=$1
PATCH_STRIDE=$2
SIGMA=$3
RMODE=$4
MAXPATCHES=$5
START_IDX=$6

module purge

singularity exec --nv \
	--overlay /scratch/aaj458/singularity_containers/my_pytorch.ext3:ro \
	/scratch/aaj458/singularity_containers/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
	/bin/bash -c "source /ext3/env.sh; python infer_certify_pretrained_salman_uncorrelated.py cifar10 -dpath /scratch/aaj458/data/Cifar10/ --alpha 0.01 -mp salman_models/pretrained_models/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_$SIGMA/checkpoint.pth.tar -mt resnet110 -ni 500 -ps $PATCH_SIZE -pstr $PATCH_STRIDE -sigma $SIGMA --N0 100 --N 10000 --patch -o cifar10_SOTA_same_numsamples/certify_results_salman_patchsmooth_smoothmax_randompatches_$MAXPATCHES/ --batch 400 -rm $RMODE -ns 36 -rp -np $MAXPATCHES --normalize -si $START_IDX"


#singularity exec --nv \
#	--overlay /scratch/aaj458/singularity_containers/my_pytorch.ext3:ro \
#	/scratch/aaj458/singularity_containers/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif \
#	/bin/bash -c "source /ext3/env.sh; python infer_certify_pretrained_salman.py cifar10 -dpath /scratch/aaj458/data/Cifar10/ -mt resnet110 -mp salman_models/pretrained_models/cifar10/finetune_cifar_from_imagenetPGD2steps/PGD_10steps_30epochs_multinoise/2-multitrain/eps_64/cifar10/resnet110/noise_0.12/checkpoint.pth.tar -ni 100 -ps $PATCH_SIZE -pstr $PATCH_STRIDE -sigma $SIGMA --N0 100 --N 10000 -o cifar10_SOTA/ --batch 400 --normalize"
