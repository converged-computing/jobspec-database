#!/bin/bash

#OAR -n MovingCompany
#OAR -l /nodes=1/gpu=1,walltime=4:00:00
#OAR --stdout %jobid%.out
#OAR --stderr %jobid%.err
#OAR --project pr-ai4cmas
#OAR -p gpumodel='V100'

cd /bettik/soulej/omarl_experiments/envs/mcy
source ./python-envs/mcy/bin/activate
python3.9 train_test.py -mode train -num_cpu 8
