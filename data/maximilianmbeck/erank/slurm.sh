#!/bin/bash -e
#SBATCH --partition=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:3 --constraint='T4'
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=beck@ml.jku.at



# use other gpu: --gres=gpu:a100-pcie-40gb:2
# activate env
eval "$(conda shell.bash hook)"
conda activate subspaces
which python

# make sure pytorch LSTMs are deterministic
# export CUBLAS_WORKSPACE_CONFIG=:16:8

# allow pytorch/numpy to use all cores
NUM_CORES=32
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

python run_sweep.py --config-name 11.7.1_mnist_lenet_rotatedtasks.yaml