#!/bin/bash
#SBATCH -J cb_v2
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH -G 2
#SBATCH --nodes 1

# You can start a container on each node from a shared squashfs file 
srun --mpi=pmix --container-image=/users/rosenbaum/aheidar/containers/tensorflow:21.07-tf2-py3.sqsh --container-save=/users/rosenbaum/aheidar/containers/tensorflow:21.07-tf2-py3.sqsh --container-mounts=/users/rosenbaum/aheidar/code/ViewFormer:/mnt/ViewFormer -w dgx02 --pty /bin/bash -c "python /mnt/ViewFormer/viewformer/cli.py train codebook --job-dir "/mnt/ViewFormer/sm7-codebook-v3" --dataset "/mnt/ViewFormer/datasets/sm7" --num-gpus 2 --batch-size 64 --n-embed 1024 --learning-rate 1.584e-3 --total-steps 200000 --num-val-workers 128 --num-workers 128"
