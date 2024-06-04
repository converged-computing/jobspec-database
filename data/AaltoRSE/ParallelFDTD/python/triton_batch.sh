#!/bin/bash -l
#SBATCH -J parallelFDTD
#SBATCH -o parallelFDTD.out
#SBATCH -t 00:10:00
#SBATCH -p gpu
##SBATCH --gres=gpu:teslak80:8
#SBATCH --constraint="volta|ampere|pascal"
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=128000

mkdir -p return

module load anaconda gcc/6.5.0 cuda/10.2.89 matlab/r2019b
source activate PFDTD
srun python testBench.py

mv ./*.log ./return/
mv ./*.hdf5 ./return/
mv ./*.out ./return/
