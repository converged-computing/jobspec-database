#!/bin/bash
#PBS -N st_large_inference
#PBS -l select=48
#PBS -l walltime=06:00:00
#PBS -q prod
#PBS -l filesystems=grand
#PBS -A datascience
#PBS -o logs/
#PBS -e logs/
#PBS -m abe
#PBS -M avasan@anl.gov

module load conda/2023-01-10-unstable
conda activate

fPATH=/grand/datascience/avasan/LargeScale_Inference_ST_RegGO/ST_Revised_Sort
echo $fPATH
cd $fPATH

echo quit | nvidia-cuda-mps-control
rm local_hostfile.01
./enable_mps_polaris.sh
    
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=16
let NDEPTH=64/$NRANKS_PER_NODE
let NTHREADS=$NDEPTH
TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_FORCE_GPU_ALLOW_GROWTH=true

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

split --lines=${NNODES} --numeric-suffixes=1 --suffix-length=2 $PBS_NODEFILE local_hostfile.

for lh in local_hostfile*
do
  echo "Launching mpiexec w/ ${lh}"
  mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --hostfile ${lh} --depth=${NDEPTH}  --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} ./set_affinity_gpu_polaris.sh python smiles_regress_transformer_run.py
  sleep 1s
done
wait
