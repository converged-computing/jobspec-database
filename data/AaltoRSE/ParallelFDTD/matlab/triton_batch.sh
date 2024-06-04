#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH -p gpu
#SBATCH -o LOG_FILE.out
#SBATCH --gres=gpu:1

# load the necessary modules
module load matlab/r2019b

# Run the job

srun matlab -nojvm -nosplash -batch "testBench()"
