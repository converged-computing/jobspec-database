#!/bin/bash
#PBS -A csc238
#PBS -N amg_poisson_3d_amg2013
#PBS -j oe
#PBS -l walltime=00:15:00,nodes=1

N=$[2**$PBS_ARRAYID]

module swap PrgEnv-pgi PrgEnv-gnu
module load cudatoolkit 
cd /lustre/atlas/proj-shared/csc238/projects/samg/benchmarks
aprun -n 1 ../src/samg 3 $N 0 | tee amg2013_result_direct_intp_poisson_3d_N_${N}.txt
aprun -n 1 ../src/samg 3 $N 1 | tee amg2013_result_std_intp_poisson_3d_N_${N}.txt
