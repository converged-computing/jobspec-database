#!/bin/bash
#SBATCH --job-name="bash"
#SBATCH -e bash.err
#SBATCH -p scavenger-gpu --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=25G

#Load the cuda module
module load CUDA/10.1
nvcc -V
nvidia-smi

source /hpc/home/jvp5/storage/venv/bin/activate

# Run 
python bash.py