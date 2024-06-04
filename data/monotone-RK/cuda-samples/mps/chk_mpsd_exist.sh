#!/bin/bash
#------- qsub option -----------
#PBS -A CCUSC
#PBS -q gpu
#PBS -l elapstim_req=00:03:00
#PBS -T openmpi
#PBS -b 1
#PBS -v NQSV_MPI_VER=4.1.5/gcc9.4.0-cuda11.8.0

#------- Program execution ----------
#--- move working dir
cd $PBS_O_WORKDIR
module load openmpi/"$NQSV_MPI_VER"
#--- set MPS environment variables
JOBID=$(echo $PBS_JOBID | sed -e 's/.*\://' -e 's/\.nqsv//')
CUDA_MPS_DIR=$PBS_O_WORKDIR/.mps.$JOBID
export CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_DIR
export CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_DIR
#--- check if mpsd is running
# Echo and check if MPS daemon is running
echo "Executing: ps -ef | grep [m]ps"
ps -ef | grep [m]ps

# Echo and start MPS control daemon
echo "Starting MPS control daemon: nvidia-cuda-mps-control -d"
nvidia-cuda-mps-control -d
sleep 2 # Give some time for MPS server to start

# Echo and check MPS daemon status after starting
echo "Executing: ps -ef | grep [m]ps after starting MPS"
ps -ef | grep [m]ps

# Echo and quit MPS control daemon
echo "Stopping MPS control daemon: echo quit | nvidia-cuda-mps-control"
echo quit | nvidia-cuda-mps-control
sleep 2 # Give some time for MPS server to stop

# Echo and check MPS daemon status after stopping
echo "Executing: ps -ef | grep [m]ps after stopping MPS"
ps -ef | grep [m]ps
