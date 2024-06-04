#!/bin/sh 
#SBATCH --job-name="kaiju"
#SBATCH -o /data/ssmith/logs/kaiju_%A_%a.out
#SBATCH -e /data/ssmith/logs/kaiju_%A_%a.err
#SBATCH --array=214
#SBATCH --mem 0
#SBATCH -c 16


module load Anaconda3
source activate microbiome

sample_name=$(basename /data2/ssmith/unmapped_fastqs/"$SLURM_ARRAY_TASK_ID"_*_1.fastq _1.fastq)


kaiju -t /data3/ssmith/ena/kaijudb/nodes.dmp \
-z 40 \
-m 11 \
-v \
-f /data3/ssmith/ena/kaijudb/kaiju_db_progenomes.fmi \
-i /data2/ssmith/unmapped_fastqs/"$sample_name"_1.fastq \
-j /data2/ssmith/unmapped_fastqs/"$sample_name"_2.fastq \
-o /data2/ssmith/kaiju_output/"$sample_name"_kaiju.out
