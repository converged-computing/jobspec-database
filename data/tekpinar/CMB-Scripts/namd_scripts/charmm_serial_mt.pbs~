#!/bin/bash
#PBS -m e
#PBS -l nodes=1:ppn=1
#PBS -l walltime=72:00:00
##PBS -M tekpinar@buffalo.edu
#PBS -o 1goj_solv_min_pbs.out
#PBS -j oe
#PBS -N 1goj_solv_min
#PBS
#
# charmm input file
#
INFILE=$HOME/ortho_1goj/inputscripts/1goj_solv_min.inp
OUTFILE=$HOME/ortho_1goj/output/1goj_solv_min.out

#
# generic setup and checks
#
# ulimit coredumpsize 0
# ulimit 
echo "check module commands:"
source $MODULESHOME/init/bash
limit coredumpsize 0
which module
module load charmm/c35b1r1-serial
#
# charmm binary
#
echo "Using CHARMMEXE = "$CHARMMEXE
# NPROCS=`cat $PBS_NODEFILE | wc -l`
#
# goto dir from which job was submitted and run
#  (allows standard output to go to PBS output file)
#
cd $PBS_O_WORKDIR

# module load mpich/intel-10.1/ch_p4/1.2.7p1

date
$CHARMMEXE <$INFILE>$OUTFILE 
date

#
# remove scratch directory
# \rm -r $PBSTMPDIR
