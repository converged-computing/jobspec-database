#!/bin/bash
#
# This is the pbs script template for the MMPI program
#
# We define the number of repitions for the data transfers, and the 
# message length of each transfer. These are parameters that are passed
# into the main mmmpi program as well as to script that extracts the
# matlab arrays
#
#PBS -l nodes=16:ppn=2
#PBS -l walltime=00:05:00
#PBS -A TG-MCA03S012
#PBS -q dque
#PBS -o mmpi_pjm.out
#PBS -j oe
#PBS -N mmpi_pjm
WORK_HOME=/home/ac/maechlin/tg_paper
cd $WORK_HOME
mpirun -v -machinefile $PBS_NODEFILE -np 32 ./mmpi
$WORK_HOME/mmpi2mat.py 32
exit 0
