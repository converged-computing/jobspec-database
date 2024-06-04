#!/bin/bash
#SBATCH -J cb_eval
#SBATCH -o %x.%j.out 
#SBATCH -e %x.%j.err
#SBATCH -G 1
#SBATCH --nodes 1 
# You can start a container on each node from a shared squashfs file 
srun --mpi=pmix --gpus=1 --container-image=/users/rosenbaum/aheidar/containers/tensorflow:21.07-tf2-py3.sqsh --container-save=/users/rosenbaum/aheidar/containers/tensorflow:21.07-tf2-py3.sqsh --container-mounts=/users/rosenbaum/aheidar/code/ViewFormer:/mnt/ViewFormer -w dgx01 --pty /bin/bash -c "python /mnt/ViewFormer/viewformer/cli.py evaluate codebook --codebook-model "/mnt/ViewFormer/sm7-codebook-v3/last.ckpt" --loader-path "/mnt/ViewFormer/datasets/sm7" --loader dataset --loader-split test --batch-size 64 --image-size 128 --num-store-images 0 --num-eval-images 1000 --job-dir "/mnt/ViewFormer/sm7-codebook-evaluation" " 
