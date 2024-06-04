#!/bin/sh
# An example for hybrid mpi-openmp job with gcc compilers and open MPI
#PBS -N pic
#PBS -M noone@mail.ustc.edu.cn
#PBS -o job.log
#PBS -e job.err
#PBS -q batch
#PBS -l walltime=10000:00:00
#PBS -l nodes=8:ppn=48

cd $PBS_O_WORKDIR
echo Begin Time `date`
echo Directory is $PWD
spack env activate warpx-3d

MPI_RANKS_PER_NODE=2
export HOSTS=`sort -u $PBS_NODEFILE | paste -s -d,`
export OMP_NUM_THREADS=$(($PBS_NUM_PPN / $MPI_RANKS_PER_NODE))
export OMP_SCHEDULE=dynamic
export OMP_PROC_BIND=true
export I_MPI_PIN_DOMAIN=omp
mpirun -hosts $HOSTS \
    -perhost $MPI_RANKS_PER_NODE \
    warpx.3d.MPI.OMP.DP.OPMD.PSATD.QED inputs

echo End Time `date`