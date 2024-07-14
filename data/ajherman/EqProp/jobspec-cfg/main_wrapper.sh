#!/bin/bash -l
module purge
module load miniconda3
source activate /vast/home/ajherman/miniconda3/envs/pytorch
python -u main.py "$@"
