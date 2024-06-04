#!/bin/bash
#------- qsub option -----------
#PBS -q SQUID-H
#PBS --group=jh220009
#PBS -N sam
#PBS -j o
#PBS -b 2
#PBS -l gpunum_job=8
#PBS -l elapstim_req=120:00:00

module load BaseGPU/2021
source /sqfs/home/z6b038/vir/py3/bin/activate

export PYTHONPATH=$PYTHONPATH:/sqfs/home/z6b038/.local/lib/python3.8/site-packages
export PYTHONPATH=$PYTHONPATH:/sqfs/home/z6b038/asdfghjkl
export PYTHONPATH=$PYTHONPATH:/sqfs/home/z6b038/common
export PYTHONPATH=$PYTHONPATH:/sqfs/home/z6b038/sam
export PYTHONPATH=$PYTHONPATH:/sqfs/home/z6b038/vit-pytorch

export http_proxy="http://ibgw1f-ib0:3128"
export https_proxy="https://ibgw1f-ib0:3128"

master_addr=$(cat $PBS_NODEFILE | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=19952
echo "MASTER_ADDR="$MASTER_ADDR:$MASTER_PORT

#export WANDB_MODE=offline

cd $PBS_O_WORKDIR

torchrun \
   --nnodes=1 \
   --nproc_per_node=8 \
   --max_restarts=0 \
   --rdzv_id=$LUSTRE_JOBSTAT \
   --rdzv_backend=c10d \
   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    12_sam_vs_ivon.py \
      --dataset IMAGENET \
      --data-path /sqfs/work/jh220009/data/ILSVRC2012 \
      --batch-size 256 \
      --val-batch-size 2048 \
      --label-smoothing 0.0 \
      --da-factor 1 \
      --model resnet50 \
      --opt ivon \
      --mc_samples 1 \
      --test_mc_samples 16 \
      --lr 0.4 \
      --epochs 90 \
      --warmup-steps 0 \
      --prior_prec 250.0 \
      --dampening 10 \
      --momentum_grad 0.9 \
      --momentum_hess 0.999
