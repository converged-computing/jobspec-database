#!/bin/sh

# pre-training
python src/pre_train.py \
--dataset-root ../dataset \
--dataset-save-dir ../dataset/dataset_saved \
--vocab-save-dir ../dataset/vocab_saved_new \
--do-fine-tune \
--project-name code2code \
--logging-file-path output \
--batch-size 8 \
--eval-batch-size 8 \
--n-epoch 10 \
--n-gpu 1 \
--fp16 \
--model-name shirin_code2code
