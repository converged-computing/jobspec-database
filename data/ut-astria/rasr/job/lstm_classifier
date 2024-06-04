#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Lonestar6 AMD Milan nodes
#
#   *** Serial Job in Normal Queue***
# 
# Last revised: May 5, 2022
#
# Notes:
#
#  -- For testing experimental lstm classifier
#----------------------------------------------------

#SBATCH -J lstm           # Job name
#SBATCH -o lstm.o%j       # Name of stdout output file
#SBATCH -e lstm.e%j       # Name of stderr error file
#SBATCH -p gpu-a100          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 05:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A MSS21024       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=carson.l@utexas.edu

# Launch serial code...
python /work/07965/clans/ls6/Spring_RASR/rasr/scripts/model_test.py          # Do not use ibrun or any other MPI launcher
