#!/bin/bash

#OAR -n Pistonball
#OAR -l /nodes=1/gpu=1,walltime=3:30:00
#OAR --stdout %jobid%.out
#OAR --stderr %jobid%.err
#OAR --project pr-ai4cmas
#OAR -p gpumodel='V100'

cd /bettik/soulej/omarl_experiments/pistonball2
source ./python-envs/pistonball2/bin/activate
python3 train.py
