#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -p gpgpu-1 
#SBATCH --gres=gpu:1 
#SBATCH --mem=10G
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# activate venv
source project_venv/bin/activate

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0
module load cuda

# run script from above

srun -v python -u src/project_lightning.py --num_nodes=1 --num_devices=8

# srun --output=/local/fs/myexec.out python src/project_lightning.py --num_nodes=2 --num_devices=1 
# sgather /local/fs/myexec.out myexec.out