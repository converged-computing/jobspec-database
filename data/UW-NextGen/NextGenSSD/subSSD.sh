#!/usr/bin/env bash

# Ron the short-list GPU queue
#SBATCH -p slurm_courtesy

## Request one CPU core from the scheduler
#SBATCH -c 1

## Request a GPU from the scheduler, we don't care what kind
#SBATCH --gres=gpu:gtx1080:1
#SBATCH -t 0-12:00 # time (D-HH:MM)

## Create a unique output file for the job
#SBATCH -o cuda_Training-%j.log

## Load CUDA into your environment
#module load cuda/9.0

module load usermods
module load user/cuda

## Load Euler's OpenCV
## It does not work
## module load opencv/3.3.1

# (1)
# you need to create ssd virtual environment first
# then activate the ssd virtual environment
source activate ssd

# (2)
# install software needed
#conda install -c anaconda --name ssd cudatoolkit>=9.2 --yes
#conda install -c anaconda --name ssd cudnn>=7.0.5 --yes
conda install --name ssd numpy --yes
#conda install --name ssd tensorflow-gpu --yes
conda install --name ssd tensorflow-gpu --yes
conda install -c anaconda --name ssd keras-gpu --yes
#conda install --name ssd keras --yes #>=2.2.0 --yes
#conda install --name ssd keras --yes
conda install -c anaconda --name ssd matplotlib --yes
conda install -c anaconda --name ssd beautifulsoup4 --yes 
conda install -c anaconda --name ssd scikit-learn --yes
conda install -c anaconda --name ssd Pillow --yes
conda install --name ssd opencv --yes
conda install --name ssd tqdm --yes
#conda install -c conda-forge --name ssd opencv --yes
#conda install --name ssd opencv
#conda install -c conda-forge opencv

# (3)

python3 train.py 
