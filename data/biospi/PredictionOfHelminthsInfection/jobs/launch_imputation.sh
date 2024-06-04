#!/bin/sh -login
### Job name
#PBS -N GOAT_IMP
# Name the files it will output
#PBS -o GOAT_IMP_console_output.stdout
#PBS -e GOAT_IMP_error_output.stderr
### Job configuration
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=1:ngpus=1:gputype=RTX2080Ti:ssd=true:mem=16gb

# Load the modules/environment
module purge
module load lang/python/anaconda/3.7-2020.02-tensorflow-2.1.0
source activate goat

# Enter the correct directory
cd /home/fo18103/PredictionOfHelminthsInfection/

# Run it
python gain_imputation_test.py

# Wait for background jobs to complete.
wait