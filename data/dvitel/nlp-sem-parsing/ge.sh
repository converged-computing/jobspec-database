#!/bin/bash -l
#All options below are recommended
#SBATCH -D /data/dvitel/semParse
#SBATCH -p Quick # run on partition general
#SBATCH --gpus=1 # 1 GPU
#SBATCH -w GPU45
conda activate semParse2
python3 /home/d/dvitel/semp/ge.py "$@"