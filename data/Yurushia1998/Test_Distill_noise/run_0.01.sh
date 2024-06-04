#!/bin/bash

# Configure the resources required
#SBATCH -p v100                                                # partition (this is the queue your job will be added to)
#SBATCH -N 1               	                                # number of nodes (no MPI, so we only use a single node)
#SBATCH -c 4              	                                # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=24:00:00                                         # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --mem=16GB                                              # specify memory required per node (here set to 16 GB)
#SBATCH -p v100
#SBATCH --gres=gpu:1 (for 1 GPU)

# Configure notifications 
#SBATCH --mail-type=END                                         # Send a notification email when the job is done (=END)
#SBATCH --mail-type=FAIL                                        # Send a notification email when the job fails (=FAIL)
set -e
set -x




CUDA_VISIBLE_DEVICES=0 python -m ieg.main --dataset=cifar10_uniform_0.8 --network_name=resnet29 --probe_dataset_hold_ratio=0.002 --checkpoint_path=ieg/checkpoints/ieg 

