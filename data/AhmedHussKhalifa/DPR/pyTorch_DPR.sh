#!/bin/bash
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-00:20:00
#SBATCH --output=%N-%j.out
#SBATCH --gres=gpu:p100:1

module load python/3.6.3
# source /home/ahamsala/torch_DPR/bin/activate
virtualenv --no-download $SLURM_TMPDIR/torch_DPR
source $SLURM_TMPDIR/torch_DPR/bin/activate
# module load python/3.6.3
pip install torch --no-index
pip install --no-index torch torchvision torchtext torchaudio
pip install --no-index transformers
pip install spacy[cuda] --no-index
# module load gcc/5.4.0 cuda/9
module load nixpkgs/16.09  gcc/7.3.0  cuda/10.1
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

module load python/3.6.3
module load faiss/1.6.2
deactivate
# source /home/ahamsala/torch_DPR/bin/activate
source $SLURM_TMPDIR/torch_DPR/bin/activate
python --version

# pip install wget
# python data/download_data.py --resource data.wikipedia_split.psgs_w100
# python data/download_data.py --resource data.retriever.nq
# python data/download_data.py --resource data.retriever.qas.nq

# cd data/retriever
# wget https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv -O nq-test.qa.csv

# python -m torch.distributed.launch \
# 	--nproc_per_node=1 train_dense_encoder.py \
# 	--max_grad_norm 2.0 \
# 	--encoder_model_type hf_bert \
# 	--pretrained_model_cfg bert-base-uncased \
# 	--seed 12345 \
# 	--sequence_length 256 \
# 	--warmup_steps 1237 \
# 	--batch_size 2 \
# 	--do_lower_case \
# 	--train_file "/home/ahamsala/projects/def-ehyangit/ahamsala/DPR/data/retriever/nq-train-subset.json" \
# 	--dev_file "/home/ahamsala/projects/def-ehyangit/ahamsala/DPR/data/retriever/nq-dev.json" \
# 	--output_dir "/home/ahamsala/projects/def-ehyangit/ahamsala/DPR/output" \
# 	--learning_rate 2e-05 \
# 	--num_train_epochs 1 \
# 	--dev_batch_size 16 \
#  	--val_av_rank_start_epoch 1


# python -m torch.distributed.launch \
# 	--nproc_per_node=1 train_dense_encoder.py \
# 	--max_grad_norm 2.0 \
# 	--encoder_model_type hf_bert \
# 	--pretrained_model_cfg bert-base-uncased \
# 	--seed 12345 \
# 	--sequence_length 256 \
# 	--warmup_steps 1237 \
# 	--batch_size 2 \
# 	--do_lower_case \
# 	--train_file "/home/ahamsala/projects/def-ehyangit/ahamsala/DPR/data/MSMARCO.dev.json" \
# 	--dev_file "/home/ahamsala/projects/def-ehyangit/ahamsala/DPR/data/retriever/nq-dev.json" \
# 	--output_dir "/home/ahamsala/projects/def-ehyangit/ahamsala/DPR/output" \
# 	--learning_rate 2e-05 \
# 	--num_train_epochs 1 \
# 	--dev_batch_size 16 \
#  	--val_av_rank_start_epoch 1


# python generate_dense_embeddings.py \
# 	--model_file "/home/ahamsala/projects/def-ehyangit/ahamsala/DPR/output/dpr_biencoder.0.919" \
# 	--ctx_file "/home/ahamsala/projects/def-ehyangit/ahamsala/DPR/data/wikipedia_split/psgs_w100_subset.tsv" \
# 	--shard_id 0 \
#     --num_shards 1 \
# 	--out_file "data/inference"



python dense_retriever.py \
	--model_file "/home/ahamsala/projects/def-ehyangit/ahamsala/DPR/output/dpr_biencoder.0.919" \
	--ctx_file "/home/ahamsala/projects/def-ehyangit/ahamsala/DPR/data/wikipedia_split/psgs_w100_subset.tsv" \
	--qa_file "data/retriever/nq-test.qa.csv" \
	--encoded_ctx_file "data/inference_0.pkl" \
	--out_file "data/result" \
	--n-docs 10 \
	--validation_workers 16 \
	--batch_size 32