#!/bin/bash

#SBATCH --job-name=SHA-LSTM-D
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=60:00:00
#SBATCH --mem=10000M
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --output=training_DUTCH_%a.txt

seed=$SLURM_ARRAY_TASK_ID
fn="save/DUTCH_$seed.pt"

python -u main.py --data data/dutch/ --vocab data/dutch/vocab.txt --epochs 20 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --save $fn --log-interval 100 --seed $seed --optimizer lamb --bptt 35 --warmup 30 --lr 2e-3 --emsize 650 --nhid 2600 --nlayers 4 --batch_size 128
echo "Finetuning"
python -u main.py --data data/dutch/ --epochs 4 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --save $fn --log-interval 100 --seed $seed --optimizer lamb --bptt 35 --warmup 30 --lr 1e-3 --emsize 650 --nhid 2600 --nlayers 4 --batch_size 16 --resume $fn