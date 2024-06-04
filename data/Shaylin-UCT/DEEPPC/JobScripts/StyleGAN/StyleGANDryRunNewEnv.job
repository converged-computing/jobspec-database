#!/bin/bash
#PBS -N StyleGANDryRun
#PBS -q gpu_1
#PBS -l select=1:ncpus=9:ngpus=1
#PBS -P CSCI1142
#PBS -l walltime=12:00:00
#PBS -o /mnt/lustre/users/schetty1/StyleRuns/DryRunNewEnv.out
#PBS -e /mnt/lustre/users/schetty1/StyleRuns/DryRunNewEnv.err
#PBS -m abe
#PBS -M chtsha042@myuct.ac.za
cd /home/schetty1/ 
module load chpc/python/anaconda/3-2021.11
eval "$(conda shell.bash hook)"
conda activate /home/schetty1/.conda/envs/CondaGANforStyle
module load chpc/cuda/11.6/PCIe/11.6
python3 /home/schetty1/stylegan2-ada-pytorch-main/train.py --outdir=/home/schetty1/lustre/GeneratedImages/StyleGAN2ADA/Elbow --data=/home/schetty1/lustre/ImagesforResearch/ElbowLATStyle --gpus=1 --dry-run
conda deactivate
