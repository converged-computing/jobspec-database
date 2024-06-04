#!/bin/bash
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=4:mem=50gb:ngpus=1:cl_zia=True
#PBS -N xlmr-distill

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
DATADIR=/storage/plzen1/home/barticka/rembert-test
PYTHON_SCRIPT=distill.py

singularity shell $CONTAINER
module add conda-modules
conda activate torch

cd $DATADIR

wandb login --relogin xxx

python $PYTHON_SCRIPT --lr $LR --student_model $STUDENT_MODEL --teacher_model $TEACHER_MODEL --temperature $TEMPERATURE --batch_size 4 --epochs 3
