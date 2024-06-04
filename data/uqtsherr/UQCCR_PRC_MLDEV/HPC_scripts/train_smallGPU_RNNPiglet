#!/usr/bin/env bash

#SBATCH -o t_smallGPU_out.txt
#SBATCH -e t_smallGPU_error.txt
#SBATCH --partition=gpu
#SBATCH --mem=50000
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:2
#SBATCH --job-name=TS_train_Piglet_EEG_Model
#SBATCH --time=1440

module load gnu7
module load cuda/11.0.2.450
module load anaconda
module load mvapich2
module load pmix/2.2.2

source activate timsTF

srun --mpi=pmi2 python ~/pycharmFwd/TF1\ work/RNNforPigletEEG.py
