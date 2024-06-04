#!/bin/bash
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-06:0:00
#SBATCH --output=%N-%j.out
#SBATCH --gres=gpu:p100:1


# Change to where you make your virtual environment
module load python/3.6.3
source virtual_DPR/bin/activate

# mkdir data/checkpoint
# wget https://dl.fbaipublicfiles.com/dpr/checkpoint/retriver/multiset/hf_bert_base.cp data/checkpoint


mkdir data/embedding_1



time python generate_dense_embeddings.py \
  --model_file 'data/checkpoint/hf_bert_base.cp' \
  --ctx_file '/ctx_file/CAR_collection_1.tsv' \
  --shard_id  1 \
  --num_shards 10\
  --batch_size 128 \
  --out_file 'data/embedding_1'
