#!/bin/bash
#SBATCH --array=1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 1
#SBATCH --mem=128G
#SBATCH -p a6000-gcondo --gres=gpu:1
#SBATCH --gres=gpu:1
#SBATCH -t 4-00:00:00
#SBATCH -o Output_Eval-%A.out
module load python/3.9.0
module load cuda/11.3.1
module load cudnn/8.2.0
source ~/envs/DynG2G/bin/activate
python3 -u Eval.py -f configs/config-1.yaml