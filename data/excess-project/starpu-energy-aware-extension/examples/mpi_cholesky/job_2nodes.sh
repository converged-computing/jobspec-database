#!/bin/sh -f

#the job name is "mpi run 2 nodes"
#PBS -N mpi2nodes
#PBS -q night

#use the complite path to the standard output files
#PBS -o /nas_home/hpcfapix/$PBS_JOBID.out
#PBS -e /nas_home/hpcfapix/$PBS_JOBID.err
#PBS -l walltime=00:20:00

#PBS -l nodes=1:node01:ppn=20+1:node03:ppn=24

#load used modules
module load compiler/intel/parallel_studio_2017
module load amd/app-sdk/3.0.124.132-GA
module load mpi/mpich/3.1-gnu-4.9.2
module load compiler/cuda/7.0
module load compiler/gnu/4.9.2
module load numlib/intel/mkl/11.1

DBKEY_FILE=/nas_home/hpcfapix/.mf/dbkey/${PBS_JOBID}
DBKEY=$(cat ${DBKEY_FILE})

ROOT=/nas_home/hpcfapix/starpu-energy-extension
LIB_MF=/opt/mf/stable/16.6/lib/
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ROOT}/bin/starpu/lib:${LIB_MF}
export LD_LIBRARY_PATH

EXECUTABLE=${ROOT}/examples/mpi_cholesky/mpi_cholesky

export MF_USER=hpcfapix
export MF_TASKID=${PBS_JOBNAME}
export MF_EXPID=${DBKEY}

export STARPU_CALIBRATE=1
export STARPU_SCHED=dmda
export STARPU_NCPU=2
export STARPU_NCUDA=0 # 0 GPU device is monitored 
export STARPU_NOPENCL=0
export STARPU_WORKER_CPUID="10 11"
echo "$( date +'%c' ) start mpi run n=4"
sleep 1
mpirun -n 4 -f /nas_home/hpcfapix/hostfile_4 ${EXECUTABLE}
sleep 1
echo "$( date +'%c' ) end mpi run n=4"

export STARPU_NCPU=4
export STARPU_WORKER_CPUID="10-13"
echo "$( date +'%c' ) start mpi run n=8"
sleep 1
mpirun -n 8 -f /nas_home/hpcfapix/hostfile_8 ${EXECUTABLE}
sleep 1
echo "$( date +'%c' ) end mpi run n=8"

export STARPU_NCPU=8
export STARPU_WORKER_CPUID="10-17"
echo "$( date +'%c' ) start mpi run n=16"
sleep 1
mpirun -n 16 -f /nas_home/hpcfapix/hostfile_16 ${EXECUTABLE}
sleep 1
echo "$( date +'%c' ) end mpi run n=16"

export STARPU_NCPU=16
export STARPU_WORKER_CPUID="0-15"
echo "$( date +'%c' ) start mpi run n=32"
sleep 1
mpirun -n 32 -f /nas_home/hpcfapix/hostfile_32 ${EXECUTABLE}
sleep 1
echo "$( date +'%c' ) end mpi run n=32"

#end
