#!/bin/bash

#SBATCH --job-name=train_P100

# Request CPU resource for a serial job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Request for GPU,
#SBATCH --partition=gpu
#SBATCH --gres=gpu:P100:1
# SBATCH --gres=gpu:K80:1

# Memory usage (MB)
#SBATCH --mem=10G

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=100:00:00
# SBATCH --time=00:00:20

#SBATCH --mail-user=csmi0005@student.monash.edu
#SBATCH --mail-type=FAIL

module load anaconda/5.1.0-Python3.6-gcc5
module load cudnn/7.6.5-cuda10.1
module load tensorflow/2.3.0

python Training/train.py \
    --model_name "UV_GAN_1" \
    --display_iter 50000 \
    --max_iter 500000 \
    --batch_size 1 \
    --tol 3 \
    --input "aia.np_path_normal" \
    --output "hmi.np_path_normal" \
    --connector "aia.id" "hmi.aia_id" \
# input: sql data name (table.column)
# output: sql data name (table.column)
# connector: the connection between the input and output
# tol: the time difference between input and output data in hours
