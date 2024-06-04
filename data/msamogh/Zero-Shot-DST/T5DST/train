#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 4      # cores requested
#SBATCH --mem=8gb  # memory in Mb
#SBATCH -o outfile  # send stdout to outfile
#SBATCH -e errfile  # send stderr to errfile
#SBATCH -t 20:00:00  # time requested in hour:minute:second

echo wassup
nvidia-smi
module load cuda/11.4.3
~/myblue/woz/cai-nlp/venv/bin/python T5.py --train_batch_size 2 --GPU 1 --except_domain taxi --slot_lang slottype
