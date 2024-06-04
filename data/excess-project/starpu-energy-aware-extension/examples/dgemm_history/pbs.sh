#!/bin/sh -f
#the job name is "dgemm_node01"
#PBS -N dgemm_node01
#PBS -q night

#use the complite path to the standard output files
#PBS -o /nas_home/hpcfapix/$PBS_JOBID.out
#PBS -e /nas_home/hpcfapix/$PBS_JOBID.err
#PBS -l walltime=00:60:00
#PBS -l nodes=1:node01:ppn=20

module load compiler/intel/parallel_studio_2017
module load amd/app-sdk/3.0.124.132-GA
module load mpi/mpich/3.1-gnu-4.9.2
module load compiler/cuda/7.0
module load numlib/intel/mkl/11.1
module load compiler/gnu/4.9.2

ROOT=/nas_home/hpcfapix/starpu-energy-extension

LIB=${ROOT}/src
LIB_MF=/opt/mf/stable/16.6/lib/
#PKG_CONFIG_PATH=$PKG_CONFIG_PATH:${ROOT}/bin/starpu/lib/pkgconfig
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${ROOT}/bin/starpu/lib:$LIB:${LIB_MF}
#export PKG_CONFIG_PATH
export LD_LIBRARY_PATH

DBKEY_FILE=/nas_home/hpcfapix/.mf/dbkey/${PBS_JOBID}
DBKEY=$(cat ${DBKEY_FILE})

echo "$( date +'%c' ) DBKEY is : ${DBKEY}"
#start testing on cpu
declare -a SIZE_ARRAY=(8 40 72 104 136 168 200 232)
N=1
ITER=10
EXECUTABLE=${ROOT}/examples/dgemm_history/dgemm_mf_starpu

#variables for libmfstarpu
export MF_USER=hpcfapix
export MF_TASKID=${PBS_JOBNAME}
export MF_EXPID=${DBKEY}
export STARPU_NCUDA=0 # 0 GPU device is monitored 

export STARPU_NCPU=1
export STARPU_NOPENCL=0
export STARPU_SCHED=dmda
export STARPU_CALIBRATE=1
sleep 5
for SIZE in "${SIZE_ARRAY[@]}"; do
	echo "$( date +'%c' ) [CPU] start dgemm_mf_starpu ..."
	echo "$( date +'%c' ) ${EXECUTABLE} -x ${SIZE} -y ${SIZE} -z ${SIZE} -nblocks ${N} -iter ${ITER}"
	${EXECUTABLE} -x ${SIZE} -y ${SIZE} -z ${SIZE} -nblocks ${N} -iter ${ITER}
	echo "$( date +'%c' ): ending-------------------------------------------------------------------------------"
done

#start testing on gpu
export STARPU_NCPU=0
export STARPU_NCUDA=1 # 1 GPU device is monitored
export STARPU_NOPENCL=0
export STARPU_SCHED=dmda
export STARPU_CALIBRATE=1
sleep 5
for SIZE in "${SIZE_ARRAY[@]}"; do
	echo "$( date +'%c' ): [GPU] start dgemm_mf_starpu ..."
	echo "$( date +'%c' ) ${EXECUTABLE} -x ${SIZE} -y ${SIZE} -z ${SIZE} -nblocks ${N} -iter ${ITER}"
	${EXECUTABLE} -x ${SIZE} -y ${SIZE} -z ${SIZE} -nblocks ${N} -iter ${ITER}
	echo "$( date +'%c' ): ending-------------------------------------------------------------------------------"
done
