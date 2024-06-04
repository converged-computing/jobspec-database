#!/bin/bash
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=vst14@case.edu
#SBATCH --mem=40G
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --time=100:00:00
#SBATCH -p gpu
#SBATCH -C gpup100
#SBATCH --gres=gpu:1

module purge
module load parabricks/3.1.1 singularity/3.5.1 cuda/10.1

cd /mnt/rds/txl80/LaframboiseLab/vst14/AR_BAM/

bash /mnt/rds/txl80/LaframboiseLab/vst14/pbtest_DB.sh