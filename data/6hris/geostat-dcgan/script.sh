#!/bin/sh
#SBATCH --job-name=dcgan_model

#SBATCH --mail-user=cdowdy@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200GB


#SBATCH --time=12:00:00

#SBATCH --output=job_output_%j.out

pwd; hostname; date

module load conda
conda activate pytorch-gan

export NCCL_P2P_DISABLE=1
python3 dcgan.py

date