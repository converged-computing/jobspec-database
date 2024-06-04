#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH -o "eval_logs/%x_slurm-%j.out"

# Sample usage: sbatch --job-name=eval_qald9plus-finetune_mt5-base_lc-ent noctua2/noctua2_eval.sh fine-tuned_models qald9plus-finetune_mt5-base_lc-ent datasets/qald9plus/wikidata/qald_9_plus_test_wikidata.json predictions "en" true true
set -eu
# Setting up environment
module load lib/NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0
module load ai/PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
module load vis/torchvision/0.13.1-foss-2022a-CUDA-11.7.0

source mst5-venv/bin/activate

# pass the parameters to train script
bash eval.sh "$@"


