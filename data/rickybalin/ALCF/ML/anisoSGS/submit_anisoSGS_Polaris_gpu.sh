#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N anisoSGS_scaling
#PBS -l walltime=00:10:00
#PBS -l select=4:ncpus=64:ngpus=4
#PBS -l filesystems=home:grand
#PBS -k doe
#PBS -j oe
#PBS -A datascience
#PBS -q debug-scaling
#PBS -V

EXE=scaling_anisoSGS_pt_hvd.py
MODULE=conda/2022-09-08
ENV=base

module load $MODULE 
conda activate $ENV
echo Using module $MODULE
echo and conda env $ENV
echo
module list 

cd $PBS_O_WORKDIR

NODES=$(cat $PBS_NODEFILE | wc -l)
GPUS_PER_NODE=4
RANKS=$((NODES * GPUS_PER_NODE))
echo Number of nodes: $NODES
echo Number of ranks per node: $GPUS_PER_NODE
echo Number of total ranks: $RANKS
echo

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=info 
export NCCL_NET_GDR_LEVEL=PHB
export MPICH_GPU_SUPPORT_ENABLED=1

echo mpiexec -n $RANKS --ppn $GPUS_PER_NODE --hostfile $PBS_NODEFILE python $EXE --device cuda
mpiexec -n $RANKS --ppn $GPUS_PER_NODE --hostfile $PBS_NODEFILE \
 --cpu-bind none --mem-bind none \
 ../affinity.sh \
 python $EXE --device cuda --nEpochs 40 --batch 512 --nSamples 32000 \
 --precision fp32
