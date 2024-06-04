#!/bin/bash
set -euxo pipefail

#This script is intended to generate a job script that can be submitted to a PBS manager
#To generate the job scripts
#1) Modify any of the paths/function calls/etc below to meet the analysis needs/tool used
#2) bash your_generator.sh
#Submit to the queue
#Note if the number of jobs is large, you will receive an e-mail for start/stop/fail for each job
#3) Example submission loop
#for i in {0..9}; do; qsub pbs.starscript.$i; sleep 1; done

#Build up a space sep'd array of sample/file prefix
declare -a SAMPLE=(merged_primary_colon_normal merged_primary_colon_tumor merged_primary_colon_met)

#Adjust the length of the array to iterate over to match the number of samples/files to analyze
for i in {0..2}; do
    cat > pbs.biscuitscript.$i << EOF

#PBS -l walltime=120:00:00
#PBS -l mem=200gb
#PBS -l nodes=1:ppn=40
#PBS -M dean.pettinga@vai.org
#PBS -m abe
#PBS -N biscuit

# load biscuit
module load bbc/biscuit/biscuit_0_3_12_linux_amd64

#Change to WGBS directory
cd /secondary/projects/bbc/tools/dean_workflows/biscuit_blaster

#these are directional libs
#Launch biscuit
biscuit align -M -R '@RG\tLB:hg38\tID:WGBS_${SAMPLE[i]}\tPL:Illumina\tPU:hiseq2000\tSM:${SAMPLE[i]}' \
-t 40 -b 1 \
/secondary/projects/triche/ben_projects/references/human/hg38/indexes/biscuit_gencode/hg38_PA \
${SAMPLE[i]}_1.fastq.gz ${SAMPLE[i]}_2.fastq.gz | \
samblaster -M --excludeDups --addMateTags -d ${SAMPLE[i]}.disc.hg38.sam -s ${SAMPLE[i]}.split.hg38.sam -u ${SAMPLE[i]}.interleave.fastq | \
samtools view -hbu -F 4 -q 20 | \
samtools sort -@ 8 -m 5G -o ${SAMPLE[i]}.sorted.markdup.hg38.bam -O BAM -

samtools sort -o ${SAMPLE[i]}.disc.hg38.bam -O BAM ${SAMPLE[i]}.disc.hg38.sam
samtools index ${SAMPLE[i]}.disc.hg38.bam

samtools sort -o ${SAMPLE[i]}.split.hg38.bam -O BAM ${SAMPLE[i]}.split.hg38.sam
samtools index ${SAMPLE[i]}.split.hg38.bam

pigz -p 40 ${SAMPLE[i]}.interleave.fastq

samtools index ${SAMPLE[i]}.sorted.markdup.hg38.bam

EOF
done
