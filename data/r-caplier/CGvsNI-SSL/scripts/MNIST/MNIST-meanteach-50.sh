#!/bin/bash

#OAR -n MNIST-meanteach-50
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=12:00:00
#OAR --stdout scripts_logs/MNIST-meanteach-50.out
#OAR --stderr scripts_logs/MNIST-meanteach-50.err
#OAR --project pr-cg4n6

source /applis/environments/conda.sh
conda activate CGDetection

cd ~/code/CGvsNI-SSL/src
python ./main.py --train-test --data MNIST --nb_samples_train 50000 --nb_samples_test 10000 --nb_samples_labeled 50 --img_mode L --model CNN --method MeanTeacher --max_lr 0.0002 --epochs 300 --no-verbose
