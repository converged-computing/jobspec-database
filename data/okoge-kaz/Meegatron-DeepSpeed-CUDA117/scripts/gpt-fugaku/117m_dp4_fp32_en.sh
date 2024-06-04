#!/bin/bash
#YBATCH -r a100_4
#SBATCH --job-name=gpt
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --output outputs/%j.out
#SBATCH --error errors/%j.err
. /etc/profile.d/modules.sh
module load cuda/11.7
module load cudnn/cuda-11.x/8.9.0
module load nccl/cuda-11.7/2.14.3
module load openmpi/4.0.5

# Change for multinode config
source .env/bin/activate

GPU_PER_NODE=4
NNODES=1
WORLD_SIZE=$(($GPU_PER_NODE * $NNODES))
NODE_RANK=0

MASTER_ADDR=localhost
MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

CHECKPOINT_PATH=checkpoints/gpt-fugaku/117m_dp4_fp32_en/
INPUT_PREFIX=dataset
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=dataset/wikipedia/binarized/gpt-2/en_wiki_text_document
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"

mkdir -p $CHECKPOINT_PATH
mkdir -p experiments/tensorboard

DATA_PARALLEL_SIZE=4

PIPELINE_MODEL_PARALLEL_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_PARALLEL_ARGS="--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE"
MODEL_PARALLEL_ARGS="--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE"
PARALLEL_ARGS="$MODEL_PARALLEL_ARGS $PIPELINE_PARALLEL_ARGS"

DISTRIBUTED_ARGS="--nproc_per_node $GPU_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
  pretrain_gpt.py \
  --num-layers 12 \
  --hidden-size 768 \
  --num-attention-heads 12 \
  --micro-batch-size 1 \
  --global-batch-size 4 \
  --seq-length 256 \
  --max-position-embeddings 256 \
  --train-iters 500000 \
  --lr-decay-iters 320000 \
  --save $CHECKPOINT_PATH \
  --load $CHECKPOINT_PATH \
  --data-path $DATA_PATH \
  --vocab-file $INPUT_PREFIX/$VOCAB_FILE \
  --merge-file $INPUT_PREFIX/$MERGE_FILE \
  --data-impl mmap \
  --split 949,50,1 \
  --distributed-backend nccl \
  --lr 0.00015 \
  --min-lr 1.0e-5 \
  --lr-decay-style cosine \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .01 \
  --log-interval 1 \
  --save-interval 300 \
  --eval-interval 100 \
  --eval-iters 10 \
  --checkpoint-activations \
  $PARALLEL_ARGS \
  $TENSORBOARD_ARGS \
  --log-batch-size-to-tensorboard \
  --log-validation-ppl-to-tensorboard \
  --log-timers-to-tensorboard \
  --wandb-name "gpu-en-wiki-123m_dp4_fp32"
