#!/bin/bash

#SBATCH --constraint=gpuv100x
#SBATCH --nodes=6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=100gb
#SBATCH --time=10:00:00
#SBATCH --error=/home/luberjm/pl/code/benchmarking/bw50_v.out
#SBATCH --out=/home/luberjm/pl/code/benchmarking/bw50_v.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /data/luberjm/conda/etc/profile.d/conda.sh || fail "conda load fail"
conda activate ml2 || fail "conda activate fail"
module load nccl/2.7.8_cuda11.0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest cuda
# module load NCCL/2.4.7-1-cuda.10.0
# -------------------------

# run script from above
#export SLURM_JOB_NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')
#export SLURM_NODELIST=$SLURM_JOB_NODELIST
#slurm_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
#export MASTER_ADDRESS=$(echo $slurm_nodes | cut -d' ' -f1)

srun python /home/luberjm/pl/code/adjustments.py --batch-size 32 --epochs 15 --gpus 4 --nodes 6 --workers 32 --custom-coords-file /home/luberjm/pl/code/patch_coords.data --accelerator ddp --logging-name bw50_v --train-size 500000 --test-size 33500 --enc-dim 2048 --latent-dim 1024  --resnet resnet50 --read-coords || fail "python fail"


