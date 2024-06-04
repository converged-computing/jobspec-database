#!/bin/bash

#PBS -N valence

#PBS -W group_list=xxx
#PBS -q windfall
###PBS -q standard

#PBS -l select=1:ncpus=1:mem=10gb:ngpus=1

#PBS -l walltime=5:00:0
#PBS -l cput=5:00:00
#PBS -l place=free:shared

module load singularity
cd ~/work/valence_tweet/bert/
singularity exec --cleanenv --nv /extra/liuhao16/work/tf1.12_201904.img python extract_features.py \
  --input_file=$HOME/work/valence_tweet/data/test_tweets.txt \
  --output_file=$HOME/work/valence_tweet/data/test_tweets.npz \
  --vocab_file=$HOME/work/valence_tweet/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=$HOME/work/valence_tweet/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=$HOME/work/valence_tweet/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --layers=-1 \
  --max_seq_length=128 \
  --batch_size=64
