#!/bin/bash
#SBATCH -o job_logs/train_script.sh.log-%j-%a 
#SBATCH --gres=gpu:volta:1
#SBATCH -c 5
#SBATCH -a 1-4
echo test1
# Loading the required module
module load anaconda/2020b cuda/10.2 nccl/2.5.6-cuda10.2
echo test2
# Load conda environment
# conda activate disentangling-vae
echo test3
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Number of Tasks: " $SLURM_ARRAY_TASK_COUNT

# Run script
python main.py btcvae_cardamage_128 -d cardamagemedium -b 64 -l btcvae --lr 0.001 -e 200 --checkpoint-every 10 --task-id $SLURM_ARRAY_TASK_ID --task-count $SLURM_ARRAY_TASK_COUNT
