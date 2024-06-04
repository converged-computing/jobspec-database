#!/bin/bash 
#SBATCH -t 48:30:00
#SBATCH -N 1
#SBATCH -p gpu_titanrtx
# Loading modules
module purge #Unload all loaded modules
module load 2019
module load TensorFlow

echo Running on Lisa System

#Copy input file to scratch
#cp $HOME/$NAME "$TMPDIR"

# Execute a python program 
python3 $HOME/main.py
