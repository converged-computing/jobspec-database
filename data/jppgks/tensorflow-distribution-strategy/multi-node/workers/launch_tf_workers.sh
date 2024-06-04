#!/bin/bash

#PBS -l partition=gpu
#PBS -l nodes=2:ppn=1:gpus=4
#PBS -W x=nmatchpolicy:exactnode
#PBS -l walltime=0:10:00
#PBS -A lpt2_pilot_2018
#PBS -N tf_workers 
#PBS -j oe

# Make MPI commands available
module load OpenMPI
# Activate appropriate conda environment
. /data/leuven/319/vsc31962/miniconda3/etc/profile.d/conda.sh
conda activate py27-tf-source
# Add necessary CUDA files to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/apps/leuven/skylake/2018a/software/CUDA/9.1.85/extras/CUPTI/lib64:/apps/leuven/skylake/2018a/software/CUDA/9.1.85/lib64:$VSC_DATA/nccl_2.1.15-1+cuda9.1_x86_64/lib
# Change to appropriate directory
cd $VSC_DATA/tf-distribution-strategy/multi-node/workers

# Store ClusterSpec in CLUSTER_SPEC 
source export_CLUSTER_SPEC.sh

# Launch workers ($hosts should be available after sourcing export_CLUSTER_SPEC.sh)
mpirun -np 1 --map-by node python worker_0.py
mpirun -np 1 --map-by node python worker_1.py
