#!/bin/bash
#SBATCH --job-name=[1.0,0.0]Glacier
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o [1.0,0.0].out
#SBATCH -e [1.0,0.0].err

 

 
# Small Python packages can be installed in own home directory. Not recommended for big packages like tensorflow -> Follow instructions for pipenv below
# cluster_requirements.txt is a text file listing the required pip packages (one package per line)
#  pip3 install --user -r cluster_requirements.txt
#pip3 install pathlib
#pip3 install scikit-image

export PATH=/cluster/yr14ofit/miniconda/bin:$PATH
which python
python main.py --parameter=hyperparameters/hyperparameters_reference.yaml
