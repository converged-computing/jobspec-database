#!/bin/bash -login 
#PBS -l nodes=1:ppn=1:gpus=1,walltime=10:59:00,mem=8gb 
#PBS -N 1mk5_opp_yyy_xxx_uuu 
#PBS -l feature='gpgpu:intel16'
#PBS -j oe
qstat -f $PBS_JOBID
cd $PBS_O_WORKDIR

module load MATLAB-compiler/R2016a
./mtflex_opposite_parallel uuu yyy > out.txt
