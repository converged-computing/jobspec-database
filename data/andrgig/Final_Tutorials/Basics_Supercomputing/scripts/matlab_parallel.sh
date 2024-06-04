#!/bin/bash
#SBATCH -N 1                             # Number of requested nodes
#SBATCH --time=0:03:00                   # Max walltime
#SBATCH --job-name=Matlab_parallel       # Job submission name
#SBATCH --output=Matlab_parallel.out     # Output file name
#SBATCH --reservation=scbasics2          # Reservation name - will work during tutorial only


# Written by:	Shelley Knuth
# Date:		24 February 2014
# Updated:      15 July 2016
# Purpose: 	This script is for the Matlab tutorial.  Its purpose is 
#		to compare run times of serial and parallel Matlab code. 

# Load Matlab module
ml matlab

# Run matlab without a GUI
matlab -nodisplay -nodesktop -r "clear; num_workers=12; matlab_parallel;"

