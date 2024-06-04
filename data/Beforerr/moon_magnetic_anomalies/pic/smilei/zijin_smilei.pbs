#!/bin/sh
# An example for hybrid mpi-openmp job with intel compilers
#PBS -N pic
#PBS -M noone@mail.ustc.edu.cn
#PBS -o job.log
#PBS -e job.err
#PBS -q batch
#PBS -l walltime=10000:00:00
#PBS -l nodes=1:ppn=48
cd $PBS_O_WORKDIR
echo Begin Time `date`
echo Directory is $PWD
spack env activate smilei
export HOSTS=`sort -u $PBS_NODEFILE | paste -s -d,`
export OMP_NUM_THREADS=24
export OMP_SCHEDULE=dynamic
export MPI_NUM_PROCS=$((${PBS_NP}/${OMP_NUM_THREADS}))
mpirun --np ${MPI_NUM_PROCS} \
       --map-by socket:PE=${OMP_NUM_THREADS} \
       --bind-to core \
       smilei test.py
echo End Time `date`