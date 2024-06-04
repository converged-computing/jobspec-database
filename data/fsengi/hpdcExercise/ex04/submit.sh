#!/bin/bash

## Resource Request for One Node
#SBATCH --job-name=mmul_sequential
#SBATCH --partition=exercise_hpc   # partition (queue)
#SBATCH -t 0-00:10              # time limit: (D-HH:MM) 
#SBATCH --nodes=1              # number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --output=mmul.out     # file to collect standard output
#SBATCH --error=mmul.err      # file to collect standard errors

# nodes=(1 2)
# nodes=(2)
# Define an array of tasks per node configurations (e.g., tasks_per_node=(4 8 16))
# tasks_per_node=( 2 4 6 7 8 9 10 11 12 14 16)
# tasks_per_node=( 2 )

module load devtoolset/10 mpi/open-mpi-4.0.5

srun ./heat >> data.txt