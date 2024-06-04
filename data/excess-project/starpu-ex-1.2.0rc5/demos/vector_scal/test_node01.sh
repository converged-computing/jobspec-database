#!/bin/sh -f
#the job name is "vector_scal"
#PBS -N vector_scal
#PBS -q night

#use the complite path to the standard output files
#PBS -o /nas_home/hpcfapix/$PBS_JOBID.out
#PBS -e /nas_home/hpcfapix/$PBS_JOBID.err
#PBS -l walltime=00:60:00
#PBS -l nodes=1:node01:ppn=20
module load amd/app-sdk/3.0.124.132-GA
module load mpi/mpich/3.1-gnu-4.9.2
module load compiler/cuda/7.0
module load numlib/intel/mkl/11.1
module load compiler/gnu/4.9.2

ROOT=/nas_home/hpcfapix/MF/starpu-ex-1-2-0rc5
LIB=${ROOT}/bin/lib
LIB_MF=/opt/mf/stable/16.6/lib
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB:${LIB_MF}
EXECUTABLE=${ROOT}/demos/vector_scal/vector_scal
export LD_LIBRARY_PATH

DBKEY_FILE=/nas_home/hpcfapix/.mf/dbkey/${PBS_JOBID}
DBKEY=$(cat ${DBKEY_FILE})
echo "$( date +'%c' ) DBKEY is: ${DBKEY}"
PBS_USER=hpcfapix

export STARPU_SCHED=dmda
export STARPU_CALIBRATE=1
export STARPU_PROFILING=1
export STARPU_HISTORY_MAX_ERROR=50

export STARPU_NCPU=1
export STARPU_NCUDA=0
export STARPU_NOPENCL=0
export STARPU_POWER_PROFILING=1
export MF_USER=${PBS_USER}
export MF_TASKID=${PBS_JOBNAME}
export MF_EXPID=${DBKEY}

sleep 20
for (( n=630784000; n<1024000000; n=n+65536000 ))
do
	for (( i=0; i<20; i=i+1 ))
	do
		echo "$( date +'%c' ) [CPU] start: ${EXECUTABLE} -NX ${n}"
		${EXECUTABLE} -NX ${n}
		echo "$( date +'%c' ) [CPU] end: ${EXECUTABLE} -NX ${n}"
		sleep 5
	done
done

export STARPU_NCPU=0
export STARPU_NCUDA=1
export STARPU_NOPENCL=0
sleep 10
for (( n=630784000; n<1024000000; n=n+65536000 ))
do
	for (( i=0; i<20; i=i+1 ))
	do
		echo "$( date +'%c' ) [GPU] start: ${EXECUTABLE} -NX ${n}"
		${EXECUTABLE} -NX ${n}
		echo "$( date +'%c' ) [GPU] end: ${EXECUTABLE} -NX ${n}"
		sleep 5
	done
done

export STARPU_NCPU=0
export STARPU_NCUDA=0
export STARPU_NOPENCL=1
sleep 10
for (( n=630784000; n<1024000000; n=n+65536000 ))
do
        for (( i=0; i<20; i=i+1 ))
        do
		echo "$( date +'%c' ) [OPENCL] start: ${EXECUTABLE} -NX ${n}"
		${EXECUTABLE} -NX ${n}
		echo "$( date +'%c' ) [OPENCL] end: ${EXECUTABLE} -NX ${n}"
		sleep 5
	done
done
