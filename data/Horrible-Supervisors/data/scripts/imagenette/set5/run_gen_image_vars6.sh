#!/bin/bash

#SBATCH -G 1
#SBATCH --exclude=ice[100,102-105,107-109,110-134,137-150,152-161,165,167,186]
#SBATCH -t 0-05:00
#SBATCH --array 3551-4142%30
#SBATCH --mem 16G
#SBATCH --chdir=/home/jrick6/repos/data
#SBATCH --job-name=gen_image_var
#SBATCH --output=/home/jrick6/repos/data/logs/imagenette/set5/shard6/%x.%A.%a.out

hostname

nvidia-smi

/home/jrick6/.conda/envs/simclr/bin/python generate_image_variations.py \
    -tfp "/home/jrick6/tensorflow_datasets/imagenette_id/full-size-v2/1.0.0/imagenette-train.tfrecord-00006-of-00016" \
    -o "/home/jrick6/tensorflow_datasets/imagenette_id_variations5/full-size-v2/1.0.0/dir_imagenette-train.tfrecord-00006-of-00016/imagenette-train.tfrecord-00006-of-00016" \
    --input_id ${SLURM_ARRAY_TASK_ID}