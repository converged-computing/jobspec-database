#!/bin/bash -l
	
#SBATCH --nodes=4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=0
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu
#SBATCH --time 24:00:00

module load gcc/8.4.0-cuda
module load mvapich2/2.3.4
module load py-torch/1.6.0-cuda-openmp
module load py-h5py/2.10.0-mpi
module load py-mpi4py/3.0.3

source /home/coppey/venvs/venv_lcd/bin/activate
	
srun python train.py
