#!/bin/bash

#SBATCH --job-name=onsetsResume
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=102400
#SBATCH --gres=gpu
#SBATCH --time=9:00:00
#SBATCH --output=OutputResume1.%j

module load python3/intel/3.6.3 cuda/9.0.176 nccl/cuda9.0/2.4.2
python train.py with logdir=runs/model iterations=1000000 resume_iteration=25000
