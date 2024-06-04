#! /bin/bash
#SBATCH --job-name=gpu_example
#SBATCH -o gpu_example.out
#SBATCH -e gpu_example.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=4000
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#singularity pull docker://tensorflow/tensorflow:latest-gpu
singularity run --nv tensorflow_latest-gpu.sif python -c 'from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())'