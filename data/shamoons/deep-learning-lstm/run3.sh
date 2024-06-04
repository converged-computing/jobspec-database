#!/bin/bash
#SBATCH -N 1
#SBATCH --partition gpu
#SBATCH --gres gpu:1 
#SBATCH --qos gpu-award 

module load python/3.6.6
module load cuda/9.0

export CUDA_HOME=/usr/local/cuda-9.0
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64
export PATH=/usr/local/cuda-9.0/bin:$PATH
. /home1/ss19015/.local/share/virtualenvs/deep-learning-lstm-4kMNPENW/bin/activate
cd /gpfs/gpfs/project1/gr19002-001/shamoon/deep-learning-lstm
python Q3.py