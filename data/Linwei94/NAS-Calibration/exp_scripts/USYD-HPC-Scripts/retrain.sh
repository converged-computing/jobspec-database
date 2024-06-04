#!/bin/bash
#PBS -P ContraGAN
#PBS -l select=1:ncpus=9:ngpus=1:mem=30GB
#PBS -l walltime=20:00:00
#PBS -j oe

#Load modules
module load python/3.7.7 magma/2.5.3 openmpi-gcc/3.1.5

virtualenv --system-site-packages ~/pytorch
source ~/pytorch/bin/activate

cd "/scratch/ContraGAN/projects/NAS-Calibration"

python3.7 train.py --drop_path_prob=0 --weight_decay=5e-4 --epochs=350 --scheduler=focal --batch_size=128 --arch=$ARCH --criterion=$CRITERION --grad_clip=2 --auxloss_coef=$COEF
