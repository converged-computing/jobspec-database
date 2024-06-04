#!/bin/bash
#SBATCH -J lijie
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00

nvidia-smi

python -u train_sep.py \
--code_lang java \
--gpu --model_name sep_model \
--save_dir result_models/sep_models \
--train_code_path data/dual_data/java/train/code.original \
--train_summ_path data/dual_data/java/train/javadoc.original
