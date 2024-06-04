#!/bin/bash
#PBS -N Leopard_NCI
#PBS -P jx00
#PBS -r y
#PBS -q gpuvolta
#PBS -l storage=scratch/jx00 
#PBS -l walltime=01:30:00 
#PBS -l ncpus=24
#PBS -l ngpus=2
#PBS -l mem=200GB
#PBS -M nevikw39m110.nthu.edu.tw
#PBS -m e


###########################
#load modules for gpu support
module load cuda
module load cudnn
module load nccl
module load openmpi
#module load horovod/0.22.1

# setup conda environment 
# -- change the path to your own conda directory
source /scratch/jx00/cw2590/miniconda/etc/profile.d/conda.sh
conda init bash
conda activate leopard

# run the bechmark over 2 GPUs
# -- change the path to your own 
source /scratch/jx00/cw2590/DL-based-DNA-decoding/multi_gpus_train.sh


