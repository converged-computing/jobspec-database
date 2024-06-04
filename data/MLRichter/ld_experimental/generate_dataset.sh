#!/bin/bash
#SBATCH --account=laion
#SBATCH --partition=g40x
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH --job-name=dataset_generation

module load openmpi
module load cuda/11.7
export NCCL_PROTO=simple

export NCCL_PROTO=simple
#export AWS_EC2_METADATA_DISABLED=true
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn

export NCCL_DEBUG=info
export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

export PYTHONWARNINGS="ignore"
export CXX=g++

eval "$(conda shell.bash hook)"

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=33751
echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

#cd würstchen
srun python3 wiring_coco_inference.py $1 $2 $3
