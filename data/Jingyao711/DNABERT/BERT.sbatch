#!/bin/bash

#SBATCH --time=00:59:00
#SBATCH --gpus=a100:1
# SBATCH --nodes=1
#SBATCH --ntasks=1
# SBATCH --gpus-per-task=1
# SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=24
#SBATCH --mem=50G

#SBATCH --job-name=6mer_Non_overlaping_pretrain_4_6_BERT_test_data_1e6
#SBATCH --output=%x-%j.SLURMout
#SBATCh --mail-type=ALL
#SBATCh --mail-user=tangji19@msu.edu
#SBATCH --output=%x-%j.SLURMout

module purge
module load CUDA/11.8.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bert_shinhan
cd /mnt/home/tangji19/DNABERT/BERT

python Pretrain_6mer.py

scontrol show job $SLURM_JOB_ID