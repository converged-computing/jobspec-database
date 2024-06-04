#!/bin/sh
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ear-verbose=1
#SBATCH --ear-policy=monitoring
#SBATCH --ear=on

source $HOME/workplace/Tensorflow/TF_env/bin/activate
#load cuda and libraries
module load 2020
module load ear

#load Cuda Library
module load cuDNN/8.0.4.30-CUDA-11.0.2


SVDDfilename=$6
cd /scratch
mkdir -p svdd
cd /scratch/svdd
mkdir -p models
mkdir -p data
#copy all keras models to scratchdisk
cp $HOME/workplace/Tensorflow/SVDD/models/${SVDDfilename} models/${SVDDfilename}
cp $HOME/workplace/Tensorflow/SVDD/data/training.h5 data/training.h5
cp $HOME/workplace/Tensorflow/SVDD/data/testing.h5 data/testing.h5




export SLURM_LOADER_LOAD_NO_MPI_LIB=python
srun --ear=on python $HOME/workplace/Tensorflow/SVDD/svdd-default.py --dim $1 --hidden_layers "$2" --fixed_target $3 --iterations $4 --batch $5 --device "gpu" --precision $7 #>> log/example.txt
wait

cd /scratch
rm -r svdd

