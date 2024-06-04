#!/bin/bash

#
# PURPOSE: Sample submit script for cores = 1.
# USAGE:   qsub submit_sleep
#

# PBSPRO directives (must be above shell commands)
#PBS -q cpu_amd
#PBS -N sleep_demo                              
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=1gb 
#PBS -l walltime=0:10:00                      
#

#     *** User parameters ***
RUNDIR=CHANGE_ME
EXEFILE=CHANGE_ME/sleep_demo.sh
LOGFILE=CHANGE_ME/sleep_demo.log


# load modules
module load shared pbspro
module list

# change to run directory if not already there
cd $RUNDIR

# call executable with output/error redirection (no mpirun)
$EXEFILE &> $LOGFILE

# END
exit
