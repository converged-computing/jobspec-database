#!/bin/bash

#SBATCH --job-name=CADEC_train
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=12:00:00
#SBATCH --exclude=sh[00,10-16] 
#SBATCH --output=logslurms/cadec/slurm-%j.out
#SBATCH --error=logslurms/cadec/slurm-%j.err

echo "Running on $(hostname)"

# Load the conda module
export PATH=/opt/conda/bin:$PATH
conda info --envs

# Load the conda environment
source activate new_PGx_env

## Change Python version 
#conda install python=3.8
## Update pip before installing requirements
#pip install --upgrade pip
## Verify version
#python --version
#
## Install requirements
## conda install -c conda-forge rust
#rustc --version 
#pip install setuptools_rust # Required for tokenizers which are required for transformers
##pip install tokenizers # Required for transformers requirement
#pip install -r ./BARTNER/requirements.txt
#pip install git+https://github.com/fastnlp/fastNLP@dev
#pip install git+https://github.com/fastnlp/fitlog
python3 -m pip install torch torch

# Run training for cadec
python ./BARTNER/train.py --dataset_name CADEC

