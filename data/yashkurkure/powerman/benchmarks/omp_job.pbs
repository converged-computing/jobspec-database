#!/bin/bash
#PBS -l nodes=1:ppn=32
#PBS -l walltime=01:00:00
#PBS -q workq
#PBS -o omp_job.out
#PBS -e omp_job.err

# Change to the working directory
cd $PBS_O_WORKDIR

# Extract the number of cores requested
PPN=$(echo $PBS_NODEFILE | wc -l)

# Execute
if intel
    OMP_NUM_THREADS=$PPN /nfs_share/intel/is.S.x
if amd
    OMP_NUM_THREADS=$PPN /nfs_share/amd/is.S.x
