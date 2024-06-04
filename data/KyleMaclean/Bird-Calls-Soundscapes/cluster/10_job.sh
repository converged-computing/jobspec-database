#!/bin/bash

#SBATCH -p cs -A cs -q csug
#SBATCH -c4 --mem=8g
#SBATCH --gres gpu:2

module load nvidia/cuda-11.0
module load nvidia/cudnn-v8.0.180-forcuda11.0

papermill ./10_input_fresh.ipynb ./10_input_fresh_OUT.ipynb
