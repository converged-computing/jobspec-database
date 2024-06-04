#!/bin/bash -v
#PBS -q gpu
#PBS -N reformer
#PBS -l select=1:ncpus=1:ngpus=1:mem=20gb:scratch_local=10gb:cl_adan=True
#PBS -l walltime=24:00:00 
#PBS -j oe

module add python-3.6.2-gcc
module add python36-modules-gcc
module add cuda-10.0
module add cudnn-7.0

cd $PBS_O_WORKDIR

source scripts/venv.sh
export PYTHONPATH=/storage/plzen1/home/memduh/versetorch/venv/
export PYTHON=/storage/plzen1/home/memduh/versetorch/venv/bin/python
$PYTHON -m pip install reformer_pytorch transformers fairseq
$PYTHON src/train/rf.py
