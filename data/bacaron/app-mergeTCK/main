#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:20:00
#PBS -l vmem=30gb

module load matlab/2019a
module load mrtrix

matlab -nosplash -nodisplay -r mergeTCK

# Check for output.
[ ! -s track.tck ] && exit 1
