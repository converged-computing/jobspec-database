#!/bin/bash
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -J BFC
#SBATCH -p defq,short
#SBATCH --array=1-22
#SBATCH --mail-type=all
#SBATCH --mail-user=bnguyen@gwu.edu
#SBATCH -o out_err_files/BFC_%A_%a.out
#SBATCH -e out_err_files/BFC_%A_%a.err
name1=$(sed -n "$SLURM_ARRAY_TASK_ID"p seq_list.txt)
cd ../data/seq
module load BFC
bfc -s 3g -t 16 ${name1}_001.fastq.gz | gzip -1 > ${name1}.corrected.fastq.gz
#bfc -s 3g -t16 16_S10_L001_R1_001_run1.fastq | gzip -1 > 16_S10_L001_R1_001_run1.corrected.fq.gz
