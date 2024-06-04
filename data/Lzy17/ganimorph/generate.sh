#!/usr/bin/env bash

#SBATCH --job-name=cyclegan
#SBATCH --account=sds173
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:01:00
#SBATCH --output=pytorch-gpu-shared.o%j.%N

module purge
module list
printenv

time -p singularity exec --bind /oasis --nv /share/apps/gpu/singularity/images/pytorch/pytorch-v1.5.0-gpu-20200511.simg python3 test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --gpu_ids 0,1,2,3 --batch_size 16 --norm instance


