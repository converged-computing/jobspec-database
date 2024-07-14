#!/bin/bash

echo running slurm launcher on node: `hostname`
nvidia-smi -L

# 1. Perhaps unnecessary, but we export the relevant local rank parameters as
# environment variables.
# export RANK=$(( ($SLURM_NNODES * $SLURM_NODEID) + $SLURM_LOCALID ))
export RANK=$(echo $SLURM_GTIDS | cut -d, -f$(($SLURM_LOCALID+1)))
export WORLD_SIZE=$SLURM_NTASKS

# Note: the local rank is not really 0, but since SLURM masks the GPUs to make
# them appear as device 0, the local rank is effectively 0.
# export LOCAL_RANK=$SLURM_LOCALID
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=$SLURM_GPUS_ON_NODE

# 2. Print SLURM environment information
if [[ $RANK == 0 ]]; then
    echo Slurm job ID is $SLURM_JOB_ID
    echo This job runs on the following machines:
    echo `echo $SLURM_JOB_NODELIST | uniq`
    echo Using $SLURM_JOB_NUM_NODES nodes
    echo GPUs on node: $SLURM_GPUS_ON_NODE
fi

# 3. Set command line arguments.
ARGS="dist.world_size=$SLURM_NTASKS dist.rank=$RANK"
ARGS="$ARGS dist.master_addr=$MASTER_ADDR dist.master_port=$MASTER_PORT"
ARGS="$ARGS dist.comm_port=$COMM_PORT"

if [[ $RANK == 0 ]]; then
    echo slurm launcher launching rank: $RANK on `hostname`
    python $@ $ARGS
else
    sleep 2  # This ensures that rank 0 is created first
    echo slurm launcher launching rank: $RANK on `hostname`
    python $@ ++hydra.output_subdir=null ++hydra.run.dir=. $ARGS
fi
