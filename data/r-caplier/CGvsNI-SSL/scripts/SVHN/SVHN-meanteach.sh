#!/bin/bash

#OAR -n SVHN-meanteach
#OAR -t gpu
#OAR -l /nodes=1/gpudevice=1,walltime=12:00:00
#OAR --stdout scripts_logs/SVHN-meanteach.out
#OAR --stderr scripts_logs/SVHN-meanteach.err
#OAR --project pr-cg4n6

source /applis/environments/conda.sh
conda activate CGDetection

cd ~/code/CGvsNI-SSL/src
python ./main.py --train-test --data SVHN --nb_samples_total 60000 --nb_samples_test 10000 --nb_samples_labeled 1000 --img_mode RGB --model SimpleNet --method MeanTeacher --max_lr 0.0002 --epochs 300 --no-verbose
