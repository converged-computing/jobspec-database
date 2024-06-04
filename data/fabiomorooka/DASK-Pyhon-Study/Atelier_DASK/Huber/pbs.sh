#!/bin/bash

#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -M fabioeid.morooka@l2s.centralesupelec.fr
#PBS -m be
#PBS -N dask_seq
#PBS -j oe
#PBS -P gpi

# Load necessary modules
module load anaconda2/2019.03

# Activate anaconda environment
source activate myenv

# Move to directory where the job was submitted
cd $PBS_O_WORKDIR

# Run python script
python simple.py