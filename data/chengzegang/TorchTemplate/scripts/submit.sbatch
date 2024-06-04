#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=tandon_a100_2
#SBATCH --time=48:00:00
#SBATCH --mem=64GB 
#SBATCH --job-name=vit
#SBATCH --output=vit_%j.out
#SBATCH --mail-type=END

module purge

export NCCL_DEBUG=INFO

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
echo $nodes
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
echo $SLURM_JOB_ID
echo $head_node_ip:29500


srun singularity exec --nv \
	    --overlay /scratch/$USER/containers/overlay-50G-10M-2.ext3:ro \
		--overlay /scratch/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
  		--overlay /scratch/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
	    /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; \
		torchrun --nnodes 2 \
		--nproc_per_node 1 \
		--rdzv_id $SLURM_JOB_ID \
		--rdzv_backend c10d \
		--rdzv_endpoint $head_node_ip:29500 \
		-m project \
		train -m vit -t denoise -r /imagenet \
		--num-workers 16 --batch-size 128 --logdir logs/224 --total-epochs 600 \
		--warmup-epochs 30 --image-size 224 --patch-size 16 \
		--ddp --mnmg"
