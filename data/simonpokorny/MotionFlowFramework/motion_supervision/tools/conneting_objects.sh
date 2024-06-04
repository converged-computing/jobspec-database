#!/bin/bash
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

#SBATCH --mem=50G
#SBATCH --partition=amdgpu
#SBATCH --gres=gpu:1

#SBATCH --error=logs/%j.out
#SBATCH --output=logs/%j.out


ml PyTorch3D/0.7.1-foss-2021b-CUDA-11.4.1
ml OpenCV/4.5.5-foss-2021b-CUDA-11.4.1-contrib

cd $HOME

python -u motion_supervision/run_test.py

