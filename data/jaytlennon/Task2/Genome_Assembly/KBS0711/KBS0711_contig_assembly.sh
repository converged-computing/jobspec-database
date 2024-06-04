#!/bin/bash
#PBS -k o
#PBS -l nodes=1:ppn=8,vmem=100gb,walltime=10:00:00
#PBS -M wrshoema@indiana.edu
#PBS -m abe
#PBS -j oe
module load java
module load fastqc
module load bioperl
module load python
module load gcc
module load cutadapt
module load khmer
module load spades

cd /N/dc2/projects/muri2/Task2/GSF966/KBS0711

##### Clean fastqc files


cutadapt -q 30 -b AGATCGGAAGAGCACACGTCTGAACTCCAGTCACGTCCGCACATCTCGTA \
    -b GATCGGAAGAGCACACGTCTGAACTCCAGTCACGTCCGCACATCTCGTAT \
    --minimum-length 20 \
    -u 15 \
    -u -10 \
    -o tmp.1.fastq \
    -p tmp.2.fastq \
    GSF966-5-Jonslin-6k_S5_R1_001.fastq.gz GSF966-5-Jonslin-6k_S5_R2_001.fastq.gz

cutadapt -q 30 -b AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGTAGATCTCGGTGGTCGCC \
    --minimum-length 20 \
    -u 15 \
    -u -10 \
    -o GSF966-5-Jonslin-6k_S5_R2_001_Q30_U15_UN10.fastq.gz \
    -p GSF966-5-Jonslin-6k_S5_R1_001_Q30_U15_UN10.fastq.gz \
    tmp.2.fastq tmp.1.fastq

rm tmp.1.fastq tmp.2.fastq

# 11 kb

cutadapt -q 30 -b AGATCGGAAGAGCACACGTCTGAACTCCAGTCACGTGAAACGATCTCGTA \
    --minimum-length 20 \
    -u 15 \
    -u -10 \
    -o tmp.1.fastq \
    -p tmp.2.fastq \
    GSF966-6-Jonslin-11k_S6_R1_001.fastq.gz GSF966-6-Jonslin-11k_S6_R2_001.fastq.gz

cutadapt -q 30 -b AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGTAGATCTCGGTGGTCGCC \
    --minimum-length 20 \
    -u 15 \
    -u -10 \
    -o GSF966-6-Jonslin-11k_S6_R2_001_Q30_U15_UN10.fastq.gz \
    -p GSF966-6-Jonslin-11k_S6_R1_001_Q30_U15_UN10.fastq.gz \
    tmp.2.fastq tmp.1.fastq

rm tmp.1.fastq tmp.2.fastq



cd /N/dc2/projects/muri2/Task2/GSF-911/KBS0711

cutadapt -q 30 \
    --minimum-length 20 \
    -u 15 \
    -u -20 \
    -o tmp.1.fastq \
    -p tmp.2.fastq \
    GSF911-711_S1_L001_R1_001.fastq.gz GSF911-711_S1_L001_R2_001.fastq.gz

cutadapt -q 30 \
    --minimum-length 20 \
    -u 15 \
    -u -20 \
    -o GSF911-711_S1_L001_R2_001_Q30_U15_UN20.fastq.gz \
    -p GSF911-711_S1_L001_R1_001_Q30_U15_UN20.fastq.gz \
    tmp.2.fastq tmp.1.fastq

rm tmp.1.fastq tmp.2.fastq




#### Begin digital normalization


cd /N/dc2/projects/muri2/Task2

# 6 kb
interleave-reads.py ./GSF966/KBS0711/GSF966-5-Jonslin-6k_S5_R1_001_Q30_U15_UN10.fastq.gz \
    ./GSF966/KBS0711/GSF966-5-Jonslin-6k_S5_R2_001_Q30_U15_UN10.fastq.gz \
    -o ./GSF966/KBS0711/GSF966-5-Jonslin-6k_S5_001_Q30_U15_UN10.fastq.gz

extract-paired-reads.py ./GSF966/KBS0711/GSF966-5-Jonslin-6k_S5_001_Q30_U15_UN10.fastq.gz -d ./GSF966/KBS0711

normalize-by-median.py -k 17 -M 1e+07 \
    -p ./GSF966/KBS0711/GSF966-5-Jonslin-6k_S5_001_Q30_U15_UN10.fastq.gz.pe \
    -o ./GSF966/KBS0711/GSF966-5-Jonslin-6k_S5_001_Q30_U15_UN10.fastq.gz.pe.keep

extract-paired-reads.py ./GSF966/KBS0711/GSF966-5-Jonslin-6k_S5_001_Q30_U15_UN10.fastq.gz.pe.keep \
    -d ./GSF966/KBS0711

mv ./GSF966/KBS0711/GSF966-5-Jonslin-6k_S5_001_Q30_U15_UN10.fastq.gz.pe.keep.pe \
    ./GSF966/KBS0711/GSF966-5-Jonslin-6k_S5_001_Q30_U15_UN10.pe.keep.pe.fastq

# 11 kb

interleave-reads.py ./GSF966/KBS0711/GSF966-6-Jonslin-11k_S6_R1_001_Q30_U15_UN10.fastq.gz \
    ./GSF966/KBS0711/GSF966-6-Jonslin-11k_S6_R2_001_Q30_U15_UN10.fastq.gz \
    -o ./GSF966/KBS0711/GSF966-6-Jonslin-11k_S6_001_Q30_U15_UN10.fastq.gz

extract-paired-reads.py ./GSF966/KBS0711/GSF966-6-Jonslin-11k_S6_001_Q30_U15_UN10.fastq.gz -d ./GSF966/KBS0711

normalize-by-median.py -k 17 -M 1e+07 \
    -p ./GSF966/KBS0711/GSF966-6-Jonslin-11k_S6_001_Q30_U15_UN10.fastq.gz.pe \
    -o ./GSF966/KBS0711/GSF966-6-Jonslin-11k_S6_001_Q30_U15_UN10.fastq.gz.pe.keep

extract-paired-reads.py ./GSF966/KBS0711/GSF966-6-Jonslin-11k_S6_001_Q30_U15_UN10.fastq.gz.pe.keep \
    -d ./GSF966/KBS0711

mv ./GSF966/KBS0711/GSF966-6-Jonslin-11k_S6_001_Q30_U15_UN10.fastq.gz.pe.keep.pe \
    ./GSF966/KBS0711/GSF966-6-Jonslin-11k_S6_001_Q30_U15_UN10.pe.keep.pe.fastq

#### First assembly

spades.py --careful \
    --hqmp1-12 ./GSF966/KBS0711/GSF966-5-Jonslin-6k_S5_001_Q30_U15_UN10.pe.keep.pe.fastq \
    --hqmp2-12 ./GSF966/KBS0711/GSF966-6-Jonslin-11k_S6_001_Q30_U15_UN10.pe.keep.pe.fastq \
    -o ./reference_assemblies/Janthinobacterium_sp_KBS0711/KBS0711_6kMP_11kMP_only

# trusted assemble

interleave-reads.py ./GSF-911/KBS0711/GSF911-711_S1_L001_R1_001_Q30_U15_UN20.fastq.gz \
    ./GSF-911/KBS0711/GSF911-711_S1_L001_R2_001_Q30_U15_UN20.fastq.gz \
    -o ./GSF-911/KBS0711/GSF911-711_S1_L001_001_Q30_U15_UN20.fastq.gz

extract-paired-reads.py ./GSF-911/KBS0711/GSF911-711_S1_L001_001_Q30_U15_UN20.fastq.gz -d ./GSF-911/KBS0711

normalize-by-median.py -k 17 -M 1e+07 \
    -p ./GSF-911/KBS0711/GSF911-711_S1_L001_001_Q30_U15_UN20.fastq.gz.pe \
    -o ./GSF-911/KBS0711/GSF911-711_S1_L001_001_Q30_U15_UN20.fastq.gz.pe.keep

extract-paired-reads.py ./GSF-911/KBS0711/GSF911-711_S1_L001_001_Q30_U15_UN20.fastq.gz.pe.keep \
    -d ./GSF-911/KBS0711

mv ./GSF-911/KBS0711/GSF911-711_S1_L001_001_Q30_U15_UN20.fastq.gz.pe.keep.pe \
    ./GSF-911/KBS0711/GSF911-711_S1_L001_001_Q30_U15_UN20.pe.keep.pe.fastq


# second assembly

spades.py --trusted-contigs \
    ./reference_assemblies/Janthinobacterium_sp_KBS0711/KBS0711_6kMP_11kMP_only/contigs.fasta \
    --pe1-12 ./GSF-911/KBS0711/GSF911-711_S1_L001_001_Q30_U15_UN20.pe.keep.pe.fastq \
    -o ./reference_assemblies/Janthinobacterium_sp_KBS0711/KBS0711_6kMP_11kMP_PE_merged
