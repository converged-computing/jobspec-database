#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 1000 # 2GB solicitados.
#SBATCH -p mlow # Partition to submit to master low prioriy queue
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written
python ../KITTI-MOTS-train-taskb.py -p mlow