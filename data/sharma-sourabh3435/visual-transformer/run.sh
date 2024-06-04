#!/bin/bash
#SBATCH --time=02:00:00          # max walltime, hh:mm:ss
#SBATCH --nodes 1                   # Number of nodes to request
#SBATCH --gpus-per-node=a100:1     # Number of GPUs per node to request
#SBATCH --tasks-per-node=1          # Number of processes to spawn per node
#SBATCH --cpus-per-task=1           # Number of CPUs per GPU
#SBATCH --mem=32G                   # Memory per node
#SBATCH --output=%x_%A-%a_%n-%t.out
                                    # %x=job-name, %A=job ID, %a=array value, %n=node rank, %t=task rank, %N=hostname
                                    # Note: You must manually create output directory "logs" before launching job.
#SBATCH --job-name=cnn-cifar100
#SBATCH --account=def-ttt                       # Use default account
#SBATCH --mail-user=sr925041@dal.ca
#SBATCH --mail-type=ALL


GPUS_PER_NODE=1
module load python/3.10.2


# Any remaining arguments will be passed through to the main script later
# (The pass-through works like *args or **kwargs in python.)
srun python ./cnn-cifar100.py \
        --gpus $GPUS_PER_NODE \
        --name "cnn-cifar100"