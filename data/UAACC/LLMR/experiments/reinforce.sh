#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=train
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

NAME=t5t0pce_rl

LLMR=/home/mrli/scratch/projects/LLMR/ckpts/t5_base_ce+t0_prompt/t5b-t0p-dd/model-004k
t5_dd=/home/mrli/projects/def-lilimou/ychao/hf/hub/t5b-dd
REWARD_MODEL_NAME=t0-3b
WS=/project/def-lilimou/ychao


DATA=$WS/data/dialogue/cleaned_dd/single-turn
CONFIG=$HF_HOME/hub/t5-base/config.json
TOKENIZER=$HF_HOME/hub/t5-base
SAVE=/home/mrli/scratch/projects/LLMR/ckpts/RL_dd_init


mkdir -p $SAVE
cp $0 $SAVE/

python reinforce.py \
  -d $DATA \
  -cn $CONFIG \
  -tn $TOKENIZER \
  -s src src \
  --max-tokens 512 \
  --num-training-steps 100000 \
  -lr 1e-5 \
  --num-warmup-steps 5000 \
  --iter-per-update 16 \
  --save-dir $SAVE \
  --update-per-save 1000 \
  -mn $t5_dd \
  --reward-model $WS/hf/hub/$REWARD_MODEL_NAME \
  --fp32 \
  --topk 5\
  --scheduler constant\
  --update-per-sync 5000\
  --max-norm 1 \
  --softmax \
  --entropy 0.1 \
  --denom 100 \
  --reward-clip 1 \
  --template templates/dialogue_T0.txt \
  --update-per-save 1000 \
  --update-per-log 1 \
  | tee -a $SAVE/train.log
