#!/bin/bash -l
# Submission script for serial Python job
# Luis Alfredo Avendano  2021-10-22

# Run from the current directory and with current environment
#$ -V -cwd

# Ask for some time (hh:mm:ss max of 00:10:00)
# -l h_rt=07:00:00


# ASk for some GPU
# -l coproc_p100=1

# Ask for some memory (by default, 1G, without a request)
# -l node_type=40core-768G
# -l h_vmem=600G
#$ -l h_vmem=120G
# -t 2-5
# -pe smp 3
# Send emails when job starts and ends
#$ -m be

conda activate work
#which python
unset GOMP_CPU_AFFINITY
unset KMP_AFFINITY
python train_CIFAR10.py --resume --batch_size 128  --model $1 --dataset $2 --num_workers $3 --RF_level $4 --type $5 --epochs $6  --name $7 --width $8 --record $9 --resume_solution "${10}"
