#!/bin/bash

#PBS -qgpu
#PBS -lwalltime=00:10:00
#PBS -S /bin/bash
#PBS -lnodes=1:ppn=12
#PBS -lmem=250G

cd /home/lgpu0009/code
rm ./*.sh.*

source activate base
export PYTHONPATH=home/lgpu0009/code

python train_mlp_pytorch.py
