#!/bin/bash
#PBS -N KI_0.50_Ref
#PBS -l nodes=1:ppn=36:nu-g         # 1 node, 36 cores, 2 GPUs, nu-g partition
#PBS -e run.err                       # Error output
#PBS -o run.out                       # Output file name

source ~/.bashrc
source ~/.bash_profile

cd /data5/stefan/Cavity-Formation/Simulations/References/KI/0.50

export CUDA_VISIBLE_DEVICES="0,1"

python openMM.py