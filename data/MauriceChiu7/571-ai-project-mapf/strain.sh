#!/bin/sh -l

#SBATCH -A gpu

#SBATCH --mem-per-cpu 150000
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name primal-train
#SBATCH --gres=gpu:1

module purge
#module load learning/conda-5.1.0-py36-gpu
module load anaconda
module load boost/1.64.0
module load cuda/9.0.176 cudnn/cuda-9.0_7.4
#module load ml-toolkit-gpu/tensorflow/1.12.0
#module load use.own
#module load conda-env/primal-py3.6.4
source activate 571project

python -u primal_train.py

#submit a job
#run: sbatch strain.sh

# Download file from Purdue scholar to local machine
# scp remote_username@remote_host:path_to_remote_file path_to_local_file