#!/bin/bash

python main.py --img_path "../GTTS/Samples" \
--train \
--train_csv_file "../GTTS/Labels/$1_labels_train.csv" \
--val_csv_file "../GTTS/Labels/$1_labels_val.csv" \
--conv_base_lr 3e-4 \
--dense_lr 3e-3 \
--decay \
--ckpt_path ./checkpoints_$1 \
--epochs 100 \
--num_workers 8 \
--early_stopping_patience 10
