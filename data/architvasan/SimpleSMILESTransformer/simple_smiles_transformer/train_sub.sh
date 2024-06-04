#!/bin/bash
#PBS -N st_rec0
#PBS -l select=4
#PBS -l walltime=12:00:00
#PBS -q preemptable
#PBS -l filesystems=grand
#PBS -A datascience
#PBS -o logs/
#PBS -e logs/
#PBS -m abe
#PBS -M avasan@anl.gov

module load conda/2022-09-08
#conda activate #pipt
#/lus/grand/projects/datascience/avasan/envs/condav.2022.09.08

#cd /grand/datascience/avasan/Benchmarks_ST_Publication/ST_Revised_Train_multiReceptors/3CLPro_7BQY_A_1_F

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
let NDEPTH=64/$NRANKS_PER_NODE
let NTHREADS=$NDEPTH

TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"
OUT=logfile.log

#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} -env OMP_PLACES=threads -env NCCL_COLLNET_ENABLE=1 -env NCCL_NET_GDR_LEVEL=PHB python train_model.py > $OUT

mpiexec --np 8 -ppn 4 --cpu-bind verbose,list:0,1,2,3,4,5,6,7 -env NCCL_COLLNET_ENABLE=1 -env NCCL_NET_GDR_LEVEL=PHB ./set_affinity_gpu_polaris.sh python train_model.py > $OUT
