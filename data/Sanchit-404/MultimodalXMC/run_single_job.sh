#!/bin/bash
#SBATCH --job-name=A300K_6_Simple
#SBATCH --account=project_2001083
#SBATCH --partition=gpusmall
#SBATCH --time=00:10:00
#SBATCH --mem=64000
#SBATCH --cpus-per-task=10 
#SBATCH --gres=gpu:a100:1
#SBATCH --output=out/fasttest.log


module load tykky
export PATH="/scratch/project_2001083/sanchit/xc/bin:$PATH"
export PYTHONUSERBASE=/scratch/project_2001083/sanchit/xc/myenv
#--cpus-per-task means = Number of cores

echo -e "RUNNING ON:\n"
nvidia-smi
echo -e "\n\n"


python src/main.py --lr 1e-4 --epoch 50 --dataset amazontitles300k --swa --swa_warmup 4 --swa_step 3000 --batch 128 --max_len 128 --eval_step 1000 --group_y_candidate_num 2000 --group_y_candidate_topk 75 --valid  --hidden_dim 400 --group_y_group 0 
