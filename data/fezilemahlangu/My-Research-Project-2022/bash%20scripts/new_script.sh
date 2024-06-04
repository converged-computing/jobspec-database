#!/bin/bash

#SBATCH -p batch 
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -J gym 
#SBATCH -o gym.out
echo "---------------------------"
echo "Job started on" `date`
## Load the python interpreter
## module load anaconda
# PATH=$PATH:/2089676/atari-py
# PATH=$PATH:/usr/local/cuda-11.2
source ~/.bashrc ##source conda 

# conda create -n my-env
# conda create -n my_env
conda activate my_env

conda config --env --add channels conda-forge 
## HAS THE NECESSARY COMMANDS FOR INSTALLING LIBRARIES USING THE CONDA 

# conda install -c conda-forge numpy # INSTALL NUMPY
# conda install -c conda-forge scikit-learn #INSTALL sklearn
# conda install -c conda-forge tensorflow #INSTALL TENSORLFOW 
# conda install -c conda-forge zipfile36  #INSTALL ZIP
# conda install -c conda-forge csvkit #INSTALL CSV
# conda install -c conda-forge pandas #INSTALL PANDAS
# conda install -c conda-forge pillow # INSTALL PIL
# conda install pytorch torchvision torchaudio cudatoolkit=11.2 -c pytorch #INSTALL PYTORCH WITH CUDA 

#conda install -c conda-forge opencv=4.2.0 INSTALL CV2 4.2.0 is the latest version  

# conda install -c conda-forge zipfile-deflate64 #IF ZIPFILE36 DOES NOT WORK 
# conda install -c conda-forge gym-atari python=3.6 #INSTALL GYM ATARI, needs python 3.6
# conda uninstall -c conda-forge gym 
# rm -rf /home-mscluster/fmahlangu/miniconda3/pkgs/pip-22.2.2-pyhd8ed1ab_0 # removing file because of corruption in pkgs 
# conda update --force-reinstall pip-22.2.2-pyhd8ed1ab_0 
# conda install -c conda-forge gym-all
# conda install -c conda-forge atari_py


# conda install -c conda-forge gym-atari

## Execute the python script
# python TestPhase2.py

echo "---------------------------"
echo "Job ended on" `date`