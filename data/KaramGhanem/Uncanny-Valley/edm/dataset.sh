#!/bin/bash

#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=8                                # Ask for 8 CPUs
#SBATCH --gres=gpu:2                                    # Ask for 2 GPU
#SBATCH --mem=48G                                        # Ask for 48 GB of RAM
#SBATCH -o /network/scratch/k/karam.ghanem/slurm-%j.out  # Write the log on scratch


# 1. Load the required modules
module load miniconda/3 cuda/11.7

# 2. Load your environment
conda activate edm

#srun --mem=8G conda install pytorch cudatoolkit=11.7 -c pytorch -c conda-forge

#srun --mem=8G pip install blobfile 

#conda env export -p $SLURM_TMPDIR/env > ~/Diffusion/cuda117_$SLURM_JOB_ID.yml

# 3. Copy your dataset on the compute node
#cp /home/mila/k/karam.ghanem/Diffusion/cifar10_png/train /network/datasets/cifar10_train $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR

# RDV_ADDR=$(hostname)
# WORLD_SIZE=$SLURM_JOB_NUM_NODES
# # -----

# srun -l torchrun \
#    --nproc_per_node=2\
#    --nnodes=$WORLD_SIZE\
#    --rdzv_id=$SLURM_JOB_ID\
#    --rdzv_backend=c10d\
#    --rdzv_endpoint=$RDV_ADDR\
#    --standalone\
#    --nproc_per_node=1\
#     train.py --outdir=training-runs --data=/home/mila/k/karam.ghanem/scratch/datasets/cifar10/cifar10-32x32.zip --cond=1 --arch=ddpmpp --batch-gpu=32

#  python /home/mila/k/karam.ghanem/scratch/Diffusion/minDiffusion/edm/dataset_tool.py --source=/network/datasets/imagenet.var/imagenet_torchvision/train --dest=/home/mila/k/karam.ghanem/scratch/datasets/imagenet/imagenet-64x64.zip --resolution=64x64 --transform=center-crop

python fid.py ref --data=/home/mila/k/karam.ghanem/scratch/datasets/imagenet-64x64.zip --dest=/home/mila/k/karam.ghanem/scratch/datasets/imagenet-64x64.npz

# 5. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR  /network/scratch/k/karam.ghanem
