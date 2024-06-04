#!/bin/bash

# SLURM Resource Parameters

#SBATCH -n 12                       # Number of cores
#SBATCH -N 1                        # Number of nodes always 1
#SBATCH -t 3-23:00 # D-HH:MM        # Time using the nodes
#SBATCH -p a100-gpu-shared               # Partition you submit to
#SBATCH --gres=gpu:1               # GPUs
#SBATCH --mem=32G                   # Memory you need
#SBATCH --job-name=StereoCov      # Job name
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err
#SBATCH --mail-type=END             # Type of notification BEGIN/FAIL/END/ALL
#SBATCH --mail-user=satanama.ring@gmail.com
# Executable
EXE=/bin/bash

singularity exec --nv --bind /data2/datasets/wenshanw/tartan_data:/zihao/datasets:ro,/data2/datasets/yuhengq/zihao/RAFTCov:/zihao/RAFTCov /data2/datasets/yuhengq/zihao/flowformer_v1.1.sif bash /zihao/RAFTCov/script.sh