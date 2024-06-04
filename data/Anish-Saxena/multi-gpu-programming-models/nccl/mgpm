#!/bin/bash
#SBATCH --nodes=2

NPROCS=16
NPPERSOC=$(($NPROCS>>2))
echo $NPROCS $NPPERSOC
source ~/init.sh
make clean && make
rm -r hpctoolkit*
mpirun -np $NPROCS -npersocket $NPPERSOC hpcrun -e CPUTIME -e IO -e gpu=nvidia,pc -t ./jacobi
hpcstruct --gpucfg yes hpctoolkit*
hpcstruct --gpucfg yes jacobi
hpcprof -S jacobi.hpcstruct -I jacobi.cpp -I jacobi_kernels.cu -I $NCCL_HOME  hpctoolkit*
echo "===DONE==="
