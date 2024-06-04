#!/bin/bash
#PBS -N LeaveOneOutExperiment
#PBS -o output_file.log
#PBS -e error_file.err
#PBS -l nodes=4:ppn=8
#PBS -l walltime=04:00:00
#PBS -q batch
#PBS -m abe
#PBS -M your_email@example.com

# Move to the directory where the job was submitted
cd $PBS_O_WORKDIR

# Load required modules (like Python, MPI)
module load python
module load mpi

# Activate your virtual environment if you have one
source myenv/bin/activate

# Run the MPI job
mpirun -np 32 python your_script.py

# Note: Adjust the paths, module names and versions according to your HPC environment.
