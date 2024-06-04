#!/bin/sh
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ear-verbose=1
#SBATCH --ear-policy=monitoring
#SBATCH --ear=on

#load cuda and libraries
module load 2020

#load Cuda Library
module load cuDNN/8.0.4.30-CUDA-11.0.2

# activate environment 
source $HOME/PathToYourVirtualEnv/bin/activate 

# used to get advanced metrics from ear 
export SLURM_LOADER_LOAD_NO_MPI_LIB=python

srun --ear=on python $HOME/PathToYourPythonScript.py --argument $1 --argument $2 --argument $3 --device "gpu"  >> PathToWhereYouWantToSaveWhateverYourScriptPrintToConsole.txt &

wait


