#!/bin/bash
#PBS -l select=1:ncpus=2:gpu_id=0
###PBS -l place=shared
#PBS -o out.txt
#PBS -e err.txt
#PBS -N PU
cd ~/NASA/Solar_Wind_to_Kp_index 

source ~/.bashrc
conda activate parameter_update

module load cuda-11.7

# python3 train.py --arch RNN --trainer RNN --source train --preprocess_approach normalize --max_epoch 100 --description LAST
python3 pytorch_evaluate.py
# python3 pytorch_train.py
