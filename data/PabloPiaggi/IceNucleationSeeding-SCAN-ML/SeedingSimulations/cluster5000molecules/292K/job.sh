#!/bin/bash
#BSUB -P CHP115
#BSUB -W 12:00
#BSUB -nnodes 100
#BSUB -J 292K
#BSUB -o jobout.%J
#BSUB -e jobout.%J

module load cuda/10.1.243
module load gcc/7.4.0
module load cmake/3.18.2
module load ibm-wml-ce/1.6.2-3 

export OMP_NUM_THREADS=7

date
############################################################################
# Variables definition
############################################################################
lammps_exe=/ccs/home/ppiaggi/Programs/Software-deepmd-kit-1.0/lammps-git/src/lmp_mpi
cycles=1
############################################################################

############################################################################
# Run
############################################################################
if [ -e runno ] ; then
   #########################################################################
   # Restart runs
   #########################################################################
   nn=`tail -n 1 runno | awk '{print $1}'`
   jsrun -n 600 -a 1 -c 7 -g 1 -bpacked:7 $lammps_exe -sf omp -in Restart.lmp
   #########################################################################
else
   #########################################################################
   # First run
   #########################################################################
   nn=1
   # Number of partitions
   jsrun -n 600 -a 1 -c 7 -g 1 -bpacked:7 $lammps_exe -sf omp -in start.lmp
   #########################################################################
fi
############################################################################


############################################################################
# Prepare next run
############################################################################
# Back up
############################################################################
cp restart2.lmp restart2.lmp.${nn}
cp restart.lmp restart.lmp.${nn}
cp data.final data.final.${nn}
cp log.lammps log.lammps.${nn}

############################################################################
# Check number of cycles
############################################################################
mm=$((nn+1))
echo ${mm} > runno
#cheking number of cycles
if [ ${nn} -ge ${cycles} ]; then
  exit
fi
############################################################################

############################################################################
# Resubmitting again
############################################################################
bsub job.sh
############################################################################

date