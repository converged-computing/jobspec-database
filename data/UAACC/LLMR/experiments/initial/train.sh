#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=t5-ost
#SBATCH --account=rrg-lilimou
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gres=gpu:v100l:4
#SBATCH --output=/project/def-lilimou/mrli/logs/output-%j.log
#SBATCH --error=/project/def-lilimou/mrli/logs/error-%j.log

module load python/3.7

export TRANSFORMERS_OFFLINE=1

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_GID_INDEX=mlx5_0
export NCCL_DEBUG=WARN

export HF_HOME='/project/def-lilimou/ychao/hf'

NAME=t5b-t0p-ost

MODEL_NAME=t5-base
WS=/project/def-lilimou/ychao
TEMP_WS=/scratch/mrli/
DATA=/home/mrli/projects/def-lilimou/mrli/data/ost
CONFIG=$HF_HOME/hub/$MODEL_NAME/config.json
TOKENIZER=$HF_HOME/hub/$MODEL_NAME
SAVE=$TEMP_WS/projects/LLMR/ckpts/$NAME

mkdir -p $SAVE
cp $0 $SAVE/

python train.py \
  -d $DATA \
  -cn $CONFIG \
  -tn $TOKENIZER \
  -s src tgt \
  --max-tokens 4096 \
  --num-training-steps 100000 \
  -lr 7e-4 \
  --num-warmup-steps 4000 \
  --iter-per-update 2 \
  --save-dir $SAVE \
  --update-per-save 1000 \
  -mn $WS/hf/hub/$MODEL_NAME \
  --fp32 \
  --label-smoothing 0.1 \
  | tee -a $SAVE/train.log
