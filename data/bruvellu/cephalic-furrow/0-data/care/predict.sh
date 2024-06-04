#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --array=0-141%100
#SBATCH --output=logs/predict/%a.out
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Put CUDA in place
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/apps/cuda/9.0.176/lib64/:/sw/apps/cudnn/6.0/lib64/

# Define input and output directories
INPUT="3-predict/source"
OUTPUT="3-predict/restored"

# Create array of source stacks
FILES=(${INPUT}/*.tif)

# Get filename without path
FILE=`basename ${FILES[$SLURM_ARRAY_TASK_ID]}`

# Restore dataset
python3 predict.py \
--input-dir     "${INPUT}" \
--input-pattern "${FILE}" \
--input-axes    "ZYX" \
--model-basedir "1-models" \
--model-name    "proper" \
--factor        "3" \
--output-dir    "${OUTPUT}" \
--output-name   "{file_name}{file_ext}" \
--output-dtype  "uint16" \
--n-tiles       2 8 8
