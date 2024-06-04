#!/bin/bash
#SBATCH --job-name=job_project1
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=2-00:05:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:rw /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/env.sh
conda activate


python run_giga_train.py \
	--data_dir \
/home/tw2112/datas/data_upload/giga \
--bert_model \
bert \
--output_dir \
./.outs \
--log_dir \
./.logs \
--model_recover_path \
/scratch/tw2112/models/BertBase/model.bin \
--vocab_path \
/scratch/tw2112/models/BertBase/vocab.txt \
--src_file1 \
train.src.txt \
--tgt_file1 \
train.tgt.txt \
--train_batch_size \
64 \
--valid_src_file1 \
test.src.txt \
--valid_tgt_file1 \
test.tgt.txt \
--single_mode \
False \
--prefix1 \
[_giga] \
--aux_data_dir \
/home/tw2112/datas/data_upload/multinli \
--aux_src_file \
train.src.txt \
--aux_tgt_file \
train.tgt.txt \
--aux_valid_src_file \
dev.src.txt \
--aux_valid_tgt_file \
dev.tgt.txt \
--aux_prefix \
[_aux]  
"
