#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# adapted from https://github.com/NERSC/slurm-ray-cluster

echo "starting ray worker node"
ray start --address $1 --redis-password=$2
sleep infinity
