#!/bin/bash
### Run across 2 nodes
#PBS -l select=2:ncpus=40:ngpus=8:mpiprocs=8:ompthreads=5
#PBS -l walltime=0:10:00
#PBS -q dgx

### Specify project code
#PBS -P 99999999

#PBS -N mpi
#PBS -j oe


cd "$PBS_O_WORKDIR" || exit $?

# MPI in /usr/local/mpi is not integrated with PBS
#  cannot access GPU on remote nodes (see multinode.pbs for more info)
/usr/local/mpi/bin/mpirun \
 -np `cat $PBS_NODEFILE | wc -l` --hostfile $PBS_NODEFILE \
 ./nvidia-smi.sh

# Use MPI compiled with PBS support:
PATH="/home/app/dgx/openmpi-3.1.1-gnu/bin:$PATH" ; export PATH
if [ ! -x mpi-hello ] ; then mpicc -o mpi-hello mpi-hello.c ; fi
if [ ! -x mpi_nx2latbw ] ; then mpicc -o mpi_nx2latbw mpi_nx2latbw.c ; fi
echo ====
mpirun mpi-hello
echo ====
mpirun ./nvidia-smi.sh
echo ====
mpirun mpi_nx2latbw
echo ====
