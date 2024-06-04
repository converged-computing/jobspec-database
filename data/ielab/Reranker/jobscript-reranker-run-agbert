#!/bin/bash

# set a job name
#SBATCH --job-name=reranker
#################

# a file for job output, you can check job progress
#SBATCH --output=reranker_slurm_output_%j.out
#################

# a file for errors
#SBATCH --error=reranker_slurm_output_%j.err
#################

# time needed for job
#SBATCH --time=00:30:00
#################

# gpus per node
#SBATCH --gres=gpu:1
#################

# cpus per job
#SBATCH --cpus-per-task=1
#################

# number of requested nodes
#SBATCH --nodes=1
#################

# memory per node
#SBATCH --mem=10GB
#################

# slurm will send a signal this far out before it kills the job
#SBATCH --signal=USR1@300
#################


# tasks per node
#SBATCH --tasks-per-node=1
#################

module load cuda
module load tensorflow/2.5.0-py39-cuda112
source venv/bin/activate


srun python bert_reranker.py \
	--run_file examples/agask/runs/run-bm25-agask-query-test50.res \
	--collection_file examples/agask/collection/agask_collection-test50.tsv \
	--query_file /home/koo01a/scratch/agvaluate/data/queries/agask_questions-test50.csv \
	--model_name_or_path examples/agask/models/agask_model_custom_params_queries \
	--tokenizer_name_or_path examples/agask/models/agask_model_custom_params_queries \
	--batch_size 32 \
	--cut_off 500
