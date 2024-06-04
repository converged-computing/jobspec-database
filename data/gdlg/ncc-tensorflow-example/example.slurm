#!/bin/bash
#SBATCH -N 1 # Request a single node
#SBATCH -c 4 # Request four CPU cores
#SBATCH --gres=gpu # Request one gpu
#SBATCH -p res-gpu-small # Use the res-gpu-small partition
#SBATCH --qos=short # Use the short QOS
#SBATCH -t 1-0 # Set maximum walltime to 1 day
#SBATCH --job-name=example # Name of the job
#SBATCH --mem=4G # Request 4Gb of memory

# Load the global bash profile
source /etc/profile

# Load your Python environment
source env/bin/activate

# /!\ Warning: Make sure that you load the correct version of CUDA for your version of TensorFlow.
#              Otherwise TensorFlow will *NOT* use the GPU.
# http://community.dur.ac.uk/ncc.admin/tensorflow/
module load cuda/10.0-cudnn7.4

# Run the code
# -u means unbuffered stdio
python -u example.py

