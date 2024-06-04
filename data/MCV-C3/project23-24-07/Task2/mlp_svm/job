#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 2000 # 2GB solicitados.
#SBATCH -p mlow,mlow # or mlow Partition to submit to master low prioriy queue
#SBATCH --gres gpu:1 # Request for GPU, Pascales MAX 8

SAVE_DIR=$1

# Sleep to ensure that the directory is avail.
sleep 1

python mlp_and_svm.py $SAVE_DIR $2

