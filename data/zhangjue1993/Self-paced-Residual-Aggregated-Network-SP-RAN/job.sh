#!/bin/bash
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=32GB
#PBS -l walltime=48:00:00
#PBS -l wd
#PBS -l storage=scratch/po21
#PBS -m abe
#PBS -N temp
#PBS -l jobfs=100GB

module load cuda/10.1
module load cudnn/7.6.5-cuda10.1
source /home/549/jz1585/.miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate tf

cd /scratch/po21/jz1585/0118Bris/1/



python train.py --loss BSGCRF --crf_w 0.3\
         --batch_size 8 --lr 0.001 --decay_ep 30 --model FPN_RFA_3 --block 4  \
         --max_epochs 200 \
         --data_dir /scratch/po21/jz1585/ \
         --aux False\
         --self_pace False 


python train_aux_sp_lc_1.py --loss BSGCRF --crf_w 0.3\
        --batch_size 8 --lr 0.00001 --decay_ep 3000 --model FPN_RFA_3 --block 4  \
        --max_epochs 100 \
        --data_dir /scratch/po21/jz1585/ \
        --aux True\
        --self_pace True --lamda 0.04 --lamda_decay_ep 1\
        --label_update_dir /scratch/po21/jz1585/0118Bris/1/label4/ \
        --checkpoint /scratch/po21/jz1585/0118Bris/1/2021-02-28-19-12-15/model/