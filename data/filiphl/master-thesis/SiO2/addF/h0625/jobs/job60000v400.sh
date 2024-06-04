#!/bin/sh
# Job name:
#SBATCH --job-name=filip60-4
#
# Project:
#SBATCH --account=trocks 
# nn9272k

#
# Wall clock limit:
#SBATCH --time='10:00:00'
#
# Max memory usage per task:
#SBATCH --mem-per-cpu=3000M
#
# Number of tasks (MPI ranks):
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16

## Set up job environment
source /cluster/bin/jobsetup
module load intel
module load intelmpi.intel
module load python2

## Run command
mpirun -n 64 /work/users/henriasv/filip/lammps/src/lmp_intel_cpu_intelmpi -in inputScripts/system.run.60000v400
