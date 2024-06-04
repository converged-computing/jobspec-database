#!/bin/bash

#SBATCH -N 2
#SBATCH -p defq
#SBATCH -o hostname_%j.out
#SBATCH -e hostname_%j.err
#SBATCH --gres=gpu:1
module load shared keras/2.1.5 cuda91/toolkit/9.1.85 cm-ml-python3deps/2.0.0 ml-python3/tensorflow/1.10.1  openmpi3/gcc/64/3.0.0 cudnn/7.0
srun --gres=gpu:1 /home/jluu13/seniorProjectTest/seniorProject3/rgbDepthNoPrintFR.py

