#!/bin/bash

# Copyright (C) 2022 Maxime Robeyns <ez18285@bristol.ac.uk>
#
# Usage:
#       sbatch runtune.sh <run_name>

#SBATCH --job-name san_tune
#SBATCH --partition cnu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 4
#SBATCH --gpus 8
#SBATCH --time 0-02:00:00
#SBATCH --mem 64G
#SBATCH --output=san_tune.txt

cd $SLURM_SUBMIT_DIR

module load lang/python/anaconda/3.9.7-2021.12-tensorflow.2.7.0

source /user/work/ez18285/spivenv/bin/activate

host=$(hostname | cut -d '.' -f 1)
ip=$(hostname -i)
suffix='6379'
ip_head=${ip}:${suffix}
export ip_head

let "worker_num=(${SLURM_NTASKS} - 1)"
let "total_cores=${worker_num} * ${SLURM_CPUS_PER_TASK}"
# TODO: compute total_gpus?

echo "Stopping any residual ray components..."
ray stop || true
sleep 5

# Launcht he ray header node.
./tunehead.sh ${host}

# Start the worker nodes
./tuneworker.sh ${worker_num} ${host}

python optimise_sanv2.py $ip_head $1

echo "Stopping ray cluster."
ray stop
