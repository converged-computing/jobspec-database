#!/bin/bash
#PBS -N grit
#PBS -A STF006
#PBS -j oe
#PBS -o /ccs/home/rsankar/grit/workspace/titan_nightly.$PBS_JOBID.txt
#PBS -l nodes=2
#PBS -l walltime=1:00:00

source /sw/xk6/environment-modules/3.2.10.3/sles11.3_gnu4.9.0/init/bash
module list
module unload xalt
module swap PrgEnv-pgi PrgEnv-gnu
module swap gcc gcc/6.3.0
module load cudatoolkit
module load cray-hdf5
module load cmake3
module list

set -x 
echo $PBS_JOBID
date
NEXTRUNTIME=`date --date="tomorrow 1AM" +%m%d%H%M`
qsub -a $NEXTRUNTIME /ccs/home/rsankar/grit/examples/scripts/titan_nightly_job.sh

export CTEST_SITE="titan"
export CRAY_CUDA_MPS=1
export   BOOST_DIR=$HOME/mysw/titan_gcc630/boost-1.67.0
export  KOKKOS_DIR=$HOME/mysw/titan_gcc630/kokkosCuda9.1.85_d3a9419
export    SILO_DIR=$HOME/mysw/titan_gcc630/silo-4.10.2
ctest -S $HOME/grit/examples/scripts/cray_nightly_cuda.cmake -VV

date
