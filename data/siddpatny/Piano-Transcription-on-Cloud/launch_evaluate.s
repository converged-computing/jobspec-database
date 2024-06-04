#!/bin/bash

#SBATCH --job-name=evalonsets1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=51200
#SBATCH --gres=gpu
#SBATCH --time=05:00:00
#SBATCH --output=evalOutput1.%j

module load python3/intel/3.6.3 cuda/9.0.176 nccl/cuda9.0/2.4.2
python evaluate.py runs/model/model-21000.pt --save-path output/
