#!/bin/bash
#------- qsub option -----------
#PBS -A CCUSC
#PBS -q gpu
#PBS -l elapstim_req=00:10:00
#PBS -T openmpi
#PBS -b 1
#PBS -v NQSV_MPI_VER=4.1.5/gcc9.4.0-cuda11.8.0

#------- Program execution ----------
# MPSコントロールデーモンを開始
start_mps() {
    echo "Starting MPS control daemon..."
    nvidia-cuda-mps-control -d
}

# MPSコントロールデーモンを停止
stop_mps() {
    echo "Stopping MPS control daemon..."
    echo quit | nvidia-cuda-mps-control
}

#--- move working dir
cd $PBS_O_WORKDIR
module load openmpi/"$NQSV_MPI_VER"
#--- set MPS environment variables
JOBID=$(echo $PBS_JOBID | sed -e 's/.*\://' -e 's/\.nqsv//')
CUDA_MPS_DIR=$PBS_O_WORKDIR/.mps.$JOBID
export CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_DIR
export CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_DIR

# MPSなしでアプリケーションを実行
# ./run_matmul.sh
# nsys profile -o wo_mps --gpu-metrics-device=all --force-overwrite=true ./run_matmul.sh
nsys profile -o wo_mps --gpu-metrics-device=all --force-overwrite=true mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 1 -npernode 1 ./matmul 8192 naive
# ncu --target-processes all -o wo_mps --set full -f mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 1 -npernode 1 ./matmul naive
# time mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 5 -npernode 5 ./matmul naive

# # MPSありでアプリケーションを実行
# start_mps
# # ./run_matmul.sh
# # nsys profile -o w_mps --gpu-metrics-device=all --force-overwrite=true ./run_matmul.sh
# nsys profile -o w_mps --gpu-metrics-device=all --force-overwrite=true mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 1 -npernode 1 ./matmul naive
# # time mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 5 -npernode 5 ./matmul naive
# stop_mps
