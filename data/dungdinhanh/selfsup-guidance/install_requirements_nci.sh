#!/bin/bash

#PBS -l ncpus=12
#PBS -l mem=120GB
#PBS -l jobfs=200GB
#PBS -l ngpus=4
#PBS -q gpuvolta
#PBS -P li96
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/li96+scratch/li96
#PBS -l wd

module load python3/3.9.2

python -m pip install -v --no-binary :all: --user -e .

python -m pip install -v --no-binary :all: --user -r requirements.txt
# for sm86


#pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
#
#
#conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
#
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
#
#mkdir -p $CONDA_PREFIX/etc/conda/activate.d
#
#echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
#
#pip install tensorflow-gpu==2.9.2
#
#pip install sklearn
#
#python -m pip install -v --no-binary :all: --user numpy
