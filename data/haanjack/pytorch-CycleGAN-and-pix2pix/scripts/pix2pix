#!/bin/bash

BATCH_SIZE=4
NUM_GPU=8

# Specify partition
#SBATCH -p batch
 
# Request 8 GPUs
#SBATCH -G $NUM_GPU
 
# Request exclusive node access
#SBATCH --exclusive
#SBATCH --time=00:10:00

#SBATCH --output=logs/R-%j-b$BATCH_SIZE-g$NUM_GPU-%x.out

# Run nvidia-smi to show we have GPUs
# nvidia-smi
singularity exec --nv -B $(pwd):/workspace -B /raid:/raid --pwd /workspace $HOME/simg/pytorch.simg \
	python -m torch.distributed.launch --nproc_per_node=$NUM_GPU \
		train.py --dataroot $HOME/datasets/cityscape_4k \
			--name cityscape_pix2pix \
			--model pix2pix \
			--direction BtoA \
			--batch_size=$BATCH_SIZE
 
# Run some sample workload!
sleep 300
