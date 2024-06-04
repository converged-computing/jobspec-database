#!/bin/bash

# ---------------------------------------------------------------- #
#                                                                  #
#  Example job submission script for Mat3ra.com platform           #
#                                                                  #
#  Shows resource manager directives for:                          #
#                                                                  #
#    1. the name of the job                (-N)                    #
#    2. the number of nodes to be used     (-l nodes=)             #
#    3. the number of processors per node  (-l ppn=)               #
#    4. the walltime in dd:hh:mm:ss format (-l walltime=)          #
#    5. queue                              (-q) D, OR, OF, SR, SF  #
#    6. merging standard output and error  (-j oe)                 #
#    7. email about job abort, begin, end  (-m abe)                #
#    8. email address to use               (-M)                    #
#                                                                  #
#  For more info visit https://docs.mat3ra.com/jobs-cli/overview/  #
# ---------------------------------------------------------------- #

#PBS -N LAMMPS-TEST
#PBS -j oe
#PBS -l nodes=1
#PBS -l ppn=1
#PBS -l walltime=00:00:10:00
#PBS -q D
#PBS -m abe
#PBS -M info@mat3ra.com
#
## Uncomment the line below and put a desired project name, e.g.:
##      "seminar-default", or "john-projectname".
## NOTE: this is required when using organizational accounts.
## more at https://docs.mat3ra.com/jobs-cli/batch-scripts/directives/
## The job will be charged to the corresponding project.
## When commented out, the default project for user is assumed.
##PBS -A <PROJECT_NAME_IN_ACCOUNTING_SYSTEM>

# go to the job working directory
cd $PBS_O_WORKDIR

module load lammps/12Dec2018-i-14.0.3.174-impi-5.0.2.044
export LAMMPS_EXECUTABLE=lammps

## Uncomment the lines below module to enable the use of GPU-accelerated version
# export LAMMPS_EXECUTABLE=lmp
# module load lammps/2022.01.07-g-11.2.0-ompi-4.1.1-cuda-11.5
# cp $PBS_NODEFILE .exabyte/$PBS_JOBID

mpirun -np $PBS_NP ${LAMMPS_EXECUTABLE} < in.min
