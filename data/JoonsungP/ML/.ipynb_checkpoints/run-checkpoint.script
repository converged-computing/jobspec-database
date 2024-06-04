#!/bin/bash -l

#PBATCH -J downscale
#PBATCH --gres=gpu:1
#PBATCH --nodes=1
#PBATCH --ntasks-per-node=1
#PBATCH --time=24:00:00
#PBATCH -o %x.o%j
#PBATCH -e %x.e%j

export OMP_NUM_THREADS=1
which conda
conda activate py39
python train.py

exit 0
