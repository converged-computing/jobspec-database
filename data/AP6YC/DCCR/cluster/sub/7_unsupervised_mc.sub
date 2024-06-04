#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH -J DCCR-7
#SBATCH -o /home/sap625/logs/out/%j.out
#SBATCH -e /home/sap625/logs/err/%j.err
#SBATCH --mail-user=sap625@mst.edu
#SBATCH --mail-type=begin,end,fail,requeue
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2000

N_TASKS=31

# Variables, directories, etc.
PROJECT_DIR=$HOME/dev/DCCR
VENV_DIR=$HOME/.venv/DCCR
JULIA_BIN=$HOME/julia

# Date and current folder
date
ls -la

$JULIA_BIN $PROJECT_DIR/src/experiments/7_unsupervised_mc/7_unsupervised_mc.jl $N_TASKS

# Load the python module
# module load python/3.8.2

# Activate the virtual environment for the project
# source $VENV_DIR/bin/activate

# Run the gpu check script
# python $PROJECT_DIR/src/scripts/check_gpu.py

# End with echoes
echo --- END OF CUDA CHECK ---
echo All is quiet on the western front
