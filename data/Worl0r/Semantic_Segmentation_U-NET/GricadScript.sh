#!/bin/bash
#OAR -n GricadTest
#OAR -l /nodes=1/gpu=1,walltime=08:00:00
#OAR --stdout output.out
#OAR --stderr error.err
#OAR -p gpumodel='V100'
#OAR --project pr-material-acceleration
#OAR --notify mail:brice.convers@grenoble-inp.org
source /applis/environments/cuda_env.sh bigfoot 11.8
source /applis/environments/conda.sh
conda activate torch
cd ~/Semantic_Segmentation_U-NET/src
set -x
python main.py
