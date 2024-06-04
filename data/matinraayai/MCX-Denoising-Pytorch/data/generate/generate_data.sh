#!/bin/bash
#SBATCH -N 1                # Number of nodes
#SBATCH -p ce-mri         # Partition to submit to
#SBATCH --gres=gpu:a100:1   # Number of GPUs-per-node
#SBATCH --mem=100000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -t 3-00:00
#SBATCH -o generate-validation-128x128-%j.out  # %j inserts jobid
# An example script for generating data points for training and testing in both 2D and 3D.
# See generate_data.m for possible options.

source ~/modules/matlab-mcx/source
source ~/modules/nccl/nccl_2.9.8-1+cuda11.0_x86_64/source
matlab -nodesktop -nosplash -r "generate_data(2, 'train', [1e5 1e6 1e7 1e8 1e9], 500, './validation-2D-128x128', false, [0 4], [128 128], '1')"
