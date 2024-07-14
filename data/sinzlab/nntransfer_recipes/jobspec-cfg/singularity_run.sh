#!/bin/bash
set -a
source ./.env
set +a
mkdir -p /tmp/djcache
singularity instance start --nv \
                 --env "CUDA_VISIBLE_DEVICES=$1" \
                 --bind /tmp/djcache:/cache,/var/sinz-shared:/var/sinz-shared \
                 singularity_img.sif \
                 "$2" \
                 ./jupyter_run.sh

# Example: ./singularity_run.sh 1 anix_dev_GPU1