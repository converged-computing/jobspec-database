#!/bin/bash  
#SBATCH --job-name=bootstrap  # Sets the name of the job to "bootstrap".
#SBATCH --nodes=4            # Specifies that the job should use 4 compute nodes.
#SBATCH --ntasks=10          # Sets the number of tasks to be executed in parallel to 10.
#SBATCH --cpus-per-task=1    # Allocates 1 CPU per task.
#SBATCH --time=12:00:00      # Limits the job's total runtime to 12 hours.
#SBATCH --partition=research # Assigns the job to the "research" partition.
#SBATCH --output=%x.%j.out   # Specifies the file for standard output using job name and job ID.
#SBATCH --error=%x.%j.err    # Specifies the file for standard error using the same naming convention.

module purge  # Clears all loaded modules for a clean environment.

# Loads necessary modules:
module load prun
module load gnu12
module load openmpi4
module load py3-mpi4py  # Python MPI support.
module load py3-numpy  # For numerical computations.
source ~/mypython/mypython/bin/activate  # Activates a Python virtual environment.
module load cmake  # Loads the cmake module for build process management.

mpiexec -n 10 python3 OLS.py  # Executes the Python script in parallel using 10 tasks. The number of bootstrap iterations is 100 for each task. There are 1000 iterations of bootstrap iterations in total.              l
