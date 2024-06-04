#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --partition=batch
#SBATCH -J renzoCaballero
#SBATCH -o info.%J.out
#SBATCH -e info.%J.err
#SBATCH --time=50:00:00
#SBATCH --mem=60G
#SBATCH --constraint=[intel]

#OpenMP settings:
export OMP_NUM_THREADS=1

module load matlab/R2019a

#run the application:
matlab -nodisplay < iterationLambda.m
