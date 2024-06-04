#!/bin/bash
#
# This is the pbs script template for the MMPI program
#
# We define the number of repitions for the data transfers, and the 
# message length of each transfer. These are parameters that are passed
# into the main mmmpi program as well as to script that extracts the
# matlab arrays
#
#PBS -l nodes=32:ppn=1
#PBS -l walltime=00:01:59
## PBS -q main
#PBS -o $PBS_JOBID.out
#PBS -e $PBS_JOBID.err
cd $PBS_O_WORKDIR
np=$(cat $PBS_NODEFILE | wc -l)
cat $PBS_NODEFILE
mpirun -np $np -machinefile $PBS_NODEFILE ./mmpi
exit 0
