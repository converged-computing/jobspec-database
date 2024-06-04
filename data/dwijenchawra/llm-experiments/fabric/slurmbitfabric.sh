#!/bin/zsh

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=2   # This needs to match Trainer(num_nodes=...)
#SBATCH --account=euge-k
#SBATCH --partition=gilbreth-k
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=2    # This needs to match Trainer(devices=...)
#SBATCH --time=0-02:00:00
#SBATCH --job-name=fabrictraining
#SBATCH --output=slurmout/%x-%j.out
#SBATCH --error=slurmout/%x-%j.err

# this is for auto resubmit
#SBATCH --signal=SIGUSR1@90

# activate conda env
mamba activate ml

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="ib"

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0
module load cuda/12.1.0

# run script from above
srun python multnodetest.py