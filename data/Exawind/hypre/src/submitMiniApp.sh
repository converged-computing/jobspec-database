#!/bin/bash

#PBS -N phasevi
#PBS -l nodes=6:ppn=24
#PBS -l walltime=00:20:00
#PBS -A windsim
#PBS -q batched
#PBS -o out.$PBS_JOBNAME
#PBS -j oe

module purge
module use /nopt/nrel/ecom/ecp/base/modules/gcc-6.2.0/
module load gcc/6.2.0 binutils openmpi/1.10.4

export SPACK_ROOT=/projects/windsim/exawind/SharedSoftware/spack
export SPACK_EXE=${SPACK_ROOT}/bin/spack
source ${SPACK_ROOT}/share/spack/setup-env.sh
spack load binutils
spack load openmpi

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=true
export OMP_PLACES=threads

cd $PBS_O_WORKDIR
mpiexec -np 144 --bind-to core ${HOME}/code/nalu/Nalu/build_master/naluX -i phasevi2.yaml -o log.phasevi2
