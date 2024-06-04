#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=0-12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu

# Load required modules
module load Python/3.5.2-intel-2017.u2-GCC-5.4.0-CUDA8
module load Tensorflow/1.4.0-intel-2017.u2-GCC-5.4.0-CUDA8-Python-3.5.2-GPU
module load Keras/2.1.5-intel-2017.u2-GCC-5.4.0-CUDA8-Python-3.5.2-GPU
module load CUDA/8.0.44
module load matplotlib/1.5.1-intel-2017.u2-GCC-5.4.0-CUDA8-Python-3.5.2

# Launch multiple process python code
echo "Searching for mentions"
time mpiexec -n 1 python3 dnn_regress.py
