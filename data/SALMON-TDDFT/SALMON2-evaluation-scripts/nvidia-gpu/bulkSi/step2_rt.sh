#! /bin/bash
#PBS -A ${GROUP}
#PBS -q gpu
#PBS -b 1
#PBS -T intmpi
#PBS -v NQSV_MPI_VER=19.0.5
#PBS -v OMP_NUM_THREADS=12
#PBS -v OMP_SCHEDULE=static

module load intel/19.0.5 intmpi/${NQSV_MPI_VER}

SALMON=/work/`id -gn`/`id -un`/salmon2-build/intel/salmon
INPUTFILE=./Si_rt_pulse.inp
LOGFILE=./RT_output.log

cd ${PBS_O_WORKDIR}

{
  uname -a
  cat /etc/redhat-release
  cat /proc/cpuinfo | grep 'model name' | head -n1
  echo "`grep physical.id /proc/cpuinfo | sort -u | wc -l` socket"
  mpiifort --version
  mpirun -hostfile ${PBS_NODEFILE} -np 2 -perhost 2 ${SALMON} < ${INPUTFILE}
} |& tee ${LOGFILE}
