#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodes=2 # number of nodes to use, 24 p4d(e) = 192 A100 GPUs
#SBATCH --job-name=mosaicml-stable-diffusion # name of your job
#SBATCH --gpus-per-node=8 # Number of GPU per node
#SBATCH --gres=gpu:8 # number of GPU we reserve
#SBATCH --cpus-per-task=16
#SBATCH --exclusive # job has exclusive use of the resource, no sharing
#SBATCH --wait-all-nodes=1
#SBATCH --output /apps/accelerate-pcluster-distributed-training/stable-diffusion/sd_out_%j.out
#SBATCH --error /apps/accelerate-pcluster-distributed-training/stable-diffusion/sd_err_%j.err

# default variables for Enroot
: "${APPS_PATH:=/apps}"
: "${DATA_PATH:=/fsx}"

# default variables for Enroot
: "${IMAGE:=$APPS_PATH/mosaicml-stable-diffusion.sqsh}"
: "${FSX_MOUNT:=$DATA_PATH:$DATA_PATH}"

## Plenty of EFA level variables
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4d
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_PROVIDER=efa # change to eth if you want to use ENA for comparisons
export FI_EFA_ENABLE_SHM_TRANSFER=1
export NCCL_DEBUG=INFO

export WANDB_MODE=offline

declare -a ARGS=(
    --container-image $IMAGE
    --container-mounts $FSX_MOUNT
)

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH

#srun -l "${ARGS[@]}" composer --world_size 16 --base_rank 0 --master_addr $head_node_ip --master_port 29500 benchmark.py --use_ema --use_synth_data --device_train_microbatch_size 4

srun -l "${ARGS[@]}" python3 benchmark_distributed.py
