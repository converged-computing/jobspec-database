#!/usr/bin/env bash
set -e 
set -x 
CKPT_NAME=gen-KAIROS-info-simtr

rm -rf checkpoints/${CKPT_NAME}-pred 
python dlee_train_test_init.py --model=constrained-gen --ckpt_name=${CKPT_NAME}-pred \
    --load_ckpt=checkpoints/${CKPT_NAME}/4.ckpt \
    --dataset=KAIROS \
    --eval_only \
    --train_file=data/wikievents/train_info.jsonl \
    --val_file=data/wikievents/dev_info.jsonl \
    --test_file=data/wikievents/test_info_adv.jsonl \
    --train_batch_size=2\
    --eval_batch_size=1 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=5 \
    --coref_dir=data/wikievents/coref \
    --sim_train \
    --knowledge-pair-gen \

python src/accuracy_check.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
--test-file=data/wikievents/test_info.jsonl \
--dataset=KAIROS \
--coref-file=data/wikievents/coref/test.jsonlines \
--head-only \
--coref 


