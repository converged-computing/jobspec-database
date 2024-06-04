#!/bin/bash

#OAR -n Pistonball
#OAR -l /nodes=1/gpu=1,walltime=0:30:00
#OAR --stdout %jobid%.out
#OAR --stderr %jobid%.err
#OAR --project pr-ai4cmas
#OAR -p gpumodel='V100'

cd /bettik/soulej/omarl_experiments/
source ./python-envs/pistonball/bin/activate
python3 pistonball_train.py
