#!/bin/bash

#SBATCH --constraint=gpuv100x
#SBATCH --nodes=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100gb
#SBATCH --time=00:20:00
#SBATCH --error=/home/luberjm/pl/code/benchmarking/v100par_apex.out
#SBATCH --out=/home/luberjm/pl/code/benchmarking/v100par_apex.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /data/luberjm/conda/etc/profile.d/conda.sh || fail "conda load fail"
conda activate apex2 || fail "conda activate fail"
module load nccl/2.7.8_cuda11.0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest cuda
# module load NCCL/2.4.7-1-cuda.10.0
G# -------------------------

# run script from above
#export SLURM_JOB_NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')
#export SLURM_NODELIST=$SLURM_JOB_NODELIST
#slurm_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
#export MASTER_ADDRESS=$(echo $slurm_nodes | cut -d' ' -f1)

srun python /home/luberjm/pl/code/apex.py --batch-size 4 --epochs 2 --gpus 4 --nodes 4 --workers 32 --custom-coords-file /home/luberjm/pl/code/pc.data --accelerator ddp --logging-name v100x_4nodes_16gpus_16bit --train-size 1600 --test-size 400 --read-coords || fail "python fail"


#srun python experiment.py --epochs 5 --gpus 2 --patches 500 --patch-size 512 --num-workers 5 --batch-size 1  --read-coords
