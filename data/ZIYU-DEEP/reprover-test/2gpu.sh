#!/bin/bash
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=80G
#SBATCH --ntasks-per-node=2
#SBATCH --account=gts-czhang355
#SBATCH -q embers
#SBATCH -t 4:00:00
#SBATCH -C A100-80GB

# Set environment variables for NCCL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=^docker0,lo

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=12345
export WORLD_SIZE=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES))
export RANK=${SLURM_PROCID}
export LOCAL_RANK=${SLURM_LOCALID}

# Run the training script
srun python generator/main.py fit --config generator/confs/cli_lean4_random_goal_driven_tactic_ckpt_resume_2gpu.yaml

