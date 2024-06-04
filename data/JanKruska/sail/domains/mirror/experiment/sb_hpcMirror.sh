#!/bin/sh
#PBS -N SAIL_Mirror
#PBS -q hpc1
#PBS -l nodes=1:ppn=12
#PBS -l walltime=24:00:00
#PBS -l vmem=64gb

# Necessary to pseudo-revert to old memory allocation behaviour
export MALLOC_ARENA_MAX=4

# Load Java, needed for parallel computing toolbox
# java/7, java/8 no noticable difference in terms of stability
module load java/default
module load cuda/default

# Run experiment
cd $PBS_O_HOME/sail/domains/mirror/experiment/
matlab -nodisplay -nosplash -nodesktop -r "runFoamTest"
