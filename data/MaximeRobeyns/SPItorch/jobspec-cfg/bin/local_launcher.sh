#!/bin/bash

echo starting local launcher

# A launcher for single-node runs.

# Get number of workers as:
# 1. Number of GPUs allocated by SLURM, or failing that
# 2. The number of visible GPUs on the node, or if that takes too long
# 3. Set it to a manual number.
WORKERS=${SLURM_GPUS_ON_NODE:-$(timeout 5 nvidia-smi -L | wc -l)}
if [ $WORKERS == 0 ]; then WORKERS=1; fi

echo Using $WORKERS workers on `hostname`

# export MASTER_ADDR=127.0.0.1

# TODO: figure out how to set CUDA_VISIBLE_DEVICES correctly.
# export CUDA_VISIBLE_DEVICES=0

# 1. print environment information
echo Running on a single node.
echo GPUs on node: $WORKERS

for ((rank=0; rank<${WORKERS}; rank++)); do
    # 2. Perhaps unnecessary, but we export the relevant local rank parameters
    # as environment variables.
    export RANK=$rank
    export WORLD_SIZE=$WORKERS
    export LOCAL_RANK=$rank
    export LOCAL_WORLD_SIZE=$WORKERS

    # 3. Set Hydra overrides as command line arguments
    ARGS="dist.world_size=$WORLD_SIZE dist.rank=$RANK"
    ARGS="$ARGS dist.master_addr=$MASTER_ADDR dist.master_port=$MASTER_PORT"
    ARGS="$ARGS dist.comm_port=$COMM_PORT"

    if [[ $RANK == 0 ]]; then
        echo Launching master rank 0
        CUDA_VISIBLE_DEVICES=${RANK} python $@ $ARGS &
        pids[${i}]=$!
        sleep 2
    else
        echo Launching rank $RANK
        CUDA_VISIBLE_DEVICES=${RANK} python $@ ++hydra.output_subdir=null ++hydra.run.dir=. $ARGS &
        pids[${i}]=$!
    fi
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done
