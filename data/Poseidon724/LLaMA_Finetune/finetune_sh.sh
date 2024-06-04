#!/bin/bash
#SBATCH -p gpu_a100_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:30
#SBATCH --job-name="test_mistral"
#SBATCH -o logs/slurm.%j.out
#SBATCH -e logs/slurm.%j.err

#SBATCH --gres=gpu:1
#SBATCH --mail-user=f20210329@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=ALL

spack load anaconda3@2022.05
conda init bash
eval "$(conda shell.bash hook)"
conda activate /home/prajna/.conda/envs/mistral
cd /home/prajna/mistral_finetune
#cd /scratch/prajna/cjpegrp3/
# python mistral_finetune.py 
python finetune_02.py

## SBATCH -p gpu_v100_2