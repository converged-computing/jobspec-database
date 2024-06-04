#!/bin/bash

#SBATCH --partition=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --array=0-67

RC=(*.collapsed.gz)
R1=${RC[$SLURM_ARRAY_TASK_ID]%.collapsed.gz}.pair1.truncated.gz
R2=${RC[$SLURM_ARRAY_TASK_ID]%.collapsed.gz}.pair2.truncated.gz
REF=/home/rmacleod/scratch/private/shamanka2/hg38/WholeGenomeFasta/genome.fa
##REF=/mnt/shared/scratch/rmacleod/private/shamanka2/hs37d5/hs37d5.fa ##for grch37 ref

conda init bash
source ~/.bashrc
conda activate /home/rmacleod/scratch/apps/mambaforge/envs/map-etc

bwa aln -t 8 ${REF} ${RC[$SLURM_ARRAY_TASK_ID]} > ${RC[$SLURM_ARRAY_TASK_ID]%.gz}.sai
bwa aln -t 8 ${REF} ${R1} > ${R1%.truncated.gz}.sai
bwa aln -t 8 ${REF} ${R2} > ${R2%.truncated.gz}.sai

SC=${RC[$SLURM_ARRAY_TASK_ID]%.gz}.sai
S1=${R1%.truncated.gz}.sai
S2=${R2%.truncated.gz}.sai

SM=${RC[$SLURM_ARRAY_TASK_ID]:0:6}
LB=${RC[$SLURM_ARRAY_TASK_ID]:26:12}
ID=${SM}-${LB}.collapsed
ID2=${SM}-${LB}.paired

OUTC=${RC[$SLURM_ARRAY_TASK_ID]%.collapsed.gz}.collapsed.bam
OUTP=${S1%.pair1.sai}.paired.bam

bwa samse -r $(echo "@RG\tID:${ID}\tSM:${SM}\tCN:CGG\tPL:ILLUMINA\tLB:${LB}\tDS:seqcenter@sund.ku.dk") ${REF} ${SC} ${RC} | samtools sort -m 8G -@4 -o ${OUTC}

bwa sampe -r $(echo "@RG\tID:${ID2}\tSM:${SM}\tCN:CGG\tPL:ILLUMINA\tLB:${LB}\tDS:seqcenter@sund.ku.dk") ${REF} ${S1} ${S2} ${R1} ${R2} | samtools sort -m 8G -@4 -o ${OUTP}
