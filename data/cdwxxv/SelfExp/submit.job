#!/bin/bash
#SBATCH --job-name=train.job
#SBATCH --output=/home/yuhewang/projects/SelfExp/results/temp_out-%j.txt
#SBATCH --error=/home/yuhewang/projects/SelfExp/results/temp_err-%j.txt
#SBATCH --time=2-00:00
#SBATCH --mem=50000
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ai05

# Test CUDA compiler (not needed by deep learning people, we just use the python libraries)
# /cm/shared/apps/cuda11.1/toolkit/11.1.1/bin/nvcc -o saxpy /home/yuhewang/cuda_c_code/saxpy.cu && ./saxpy

# Test nvidia-smi
nvidia-smi

# Test Python conda environment

# /cm/local/apps/python37/bin/python3.7 preprocessing/store_parse_trees.py \
#       --data_dir data/  \
#       --tokenizer_name xlnet-base-cased

# /cm/local/apps/python37/bin/python3.7 preprocessing/build_concept_store.py \
#        -i data/emotions_with_parse.json \
#        -o data/ \
#        -m xlnet-base-cased \
#        -l 5

# export TOKENIZERS_PARALLELISM=false
# /cm/local/apps/python37/bin/python3.7  model/run.py --dataset_basedir data \
#                          --lr 2e-5  --max_epochs 5 \
#                          --gpus 1 \
#                          --concept_store data/concept_store.pt
                        #  --accelerator ddp

export TOKENIZERS_PARALLELISM=false
/cm/local/apps/python37/bin/python3.7  model/infer_model.py --ckpt ckpt/model_top10_debugged_retrained.ckpt \
                         --concept_map data/emotions_idx.json \
                         --dev_file data/dev_with_parse.json \
                         --paths_output_loc data/dev_output_emotions.csv
