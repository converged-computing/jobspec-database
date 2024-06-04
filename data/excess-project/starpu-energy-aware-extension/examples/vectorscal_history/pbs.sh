#!/bin/sh -f
#the job name is "vector_history_mf_starpu"
#PBS -N vector_history_mf_starpu
#PBS -q night

#use the complite path to the standard output files
#PBS -o /nas_home/hpcfapix/$PBS_JOBID.out
#PBS -e /nas_home/hpcfapix/$PBS_JOBID.err
#PBS -l walltime=00:60:00
#PBS -l nodes=1:node03:ppn=24

module load compiler/intel/parallel_studio_2017
module load amd/app-sdk/3.0.124.132-GA
module load mpi/mpich/3.1-gnu-4.9.2
module load compiler/cuda/7.0
module load numlib/intel/mkl/11.1
module load compiler/gnu/4.9.2

ROOT=/nas_home/hpcfapix/starpu-energy-extension

LIB=${ROOT}/src
LIB_MF=/opt/mf/stable/16.6/lib/
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:${ROOT}/bin/starpu/lib/pkgconfig
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ROOT}/bin/starpu/lib:$LIB:${LIB_MF}
export PKG_CONFIG_PATH
export LD_LIBRARY_PATH

DBKEY_FILE=/nas_home/hpcfapix/.mf/dbkey/${PBS_JOBID}
DBKEY=$(cat ${DBKEY_FILE})
PBS_USER=hpcfapix

echo "$( date +'%c' ) DBKEY is : ${DBKEY}"
#start testing of cpu
echo "$( date +'%c' ) Start testing of CPU... ..."
declare -a SIZE_ARRAY=(4096 6144 10240 16384 24576 40960 59392 79872 102400)
EXECUTABLE=${ROOT}/examples/vectorscal_history/vector_scal

#variables for libmfstarpu
export MF_USER=hpcfapix
export MF_TASKID=${PBS_JOBNAME}
export MF_EXPID=${DBKEY}

export STARPU_NCPU=1
export STARPU_NCUDA=0 # 0 GPU device is monitored 
export STARPU_NOPENCL=0
export STARPU_SCHED=dmda
export STARPU_CALIBRATE=1
sleep 5
for SIZE in "${SIZE_ARRAY[@]}"; do
	for i in {1..10}; do
		echo "$( date +'%c' ) [CPU] start vector scal of size ${SIZE} ..."
		echo "$( date +'%c' ) ${EXECUTABLE} -NX ${SIZE}"
		${EXECUTABLE} -NX ${SIZE}
	done
	echo " "
	echo "$( date +'%c' ): ending-------------------------------------------------------------------------------"
done

#start testing of gpu
echo "$( date +'%c' ) Start testing of GPU... ..."
export STARPU_NCPU=0
export STARPU_NCUDA=1 # 1 GPU device is monitored 
export STARPU_NOPENCL=0
export STARPU_SCHED=dmda
export STARPU_CALIBRATE=1
sleep 5
for SIZE in "${SIZE_ARRAY[@]}"; do
	for i in {1..10}; do
		echo "$( date +'%c' ) [GPU] start vector scal of size ${SIZE} ..."
		echo "$( date +'%c' ) ${EXECUTABLE} -NX ${SIZE}"
		${EXECUTABLE} -NX ${SIZE}
	done
	echo " "
	echo "$( date +'%c' ): ending-------------------------------------------------------------------------------"
done

#start testing of opencl
echo "$( date +'%c' ) Start testing of OPENCL... ..."
export STARPU_NCPU=0
export STARPU_NCUDA=0
export STARPU_NOPENCL=1
export STARPU_SCHED=dmda
export STARPU_CALIBRATE=1
sleep 5
for SIZE in "${SIZE_ARRAY[@]}"; do
	for i in {1..20}; do
		echo "$( date +'%c' ) [OPENCL] start vector scal of size ${SIZE} ..."
		echo "$( date +'%c' ) ${EXECUTABLE} -NX ${SIZE}"
		${EXECUTABLE} -NX ${SIZE}
	done
	echo " "
	echo "$( date +'%c' ): ending-------------------------------------------------------------------------------"
done
