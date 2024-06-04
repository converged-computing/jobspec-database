#!/bin/bash
#SBATCH --job-name="mpi_grayscale"
#SBATCH --output="mpi_grayscale.%j.%N.txt"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --account=isu102
#SBATCH --export=ALL
#SBATCH -t 00:10:00
#This job runs with 1 nodes, 128 cores per node for a total of 128 tasks.

module load cpu/0.15.4 gcc/10.2.0 openmpi/4.0.4


srun -n 1 ./mpi_grayscale cat.jpg comp.jpg gray.jpg 
srun -n 2 ./mpi_grayscale cat.jpg comp.jpg gray.jpg 
srun -n 5 ./mpi_grayscale cat.jpg comp.jpg gray.jpg 
srun -n 10 ./mpi_grayscale cat.jpg comp.jpg gray.jpg 
srun -n 20 ./mpi_grayscale cat.jpg comp.jpg gray.jpg 
