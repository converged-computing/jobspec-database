#!/bin/sh
#PBS -lwalltime=00:10:00
#PBS -lnodes=1:ppn=12
#PBS -lmem=250GB
module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176 
module load OpenMPI/2.1.1-GCC-6.4.0-2.28 
module load NCCL
module load modcom

export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.0.5-CUDA-9.0.176/lib64:/hpc/eb/Debian9/CUDA/9.0.176/lib64:$LD_LIBRARY_PATH
cd Assignment1
python train_convnet_pytorch.py
