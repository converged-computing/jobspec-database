#!/bin/bash

#SBATCH --job-name=rrna_intervals
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=6G
#SBATCH  --output=/globalhome/hxo752/HPC/slurm_logs/%j.out
set -eux

#loading required modules for bedtools
module load nixpkgs/16.09 
module load gcc/5.4.0
module load intel/2016.4
module load intel/2017.1
module load bedtools

GTF=/datastore/NGSF001/analysis/references/human/gencode-40/gencode.v40.annotation.gtf
OUTDIR=/globalhome/hxo752/HPC/ngsf_git_repos/NGSF-core-projects/23-1LICH-001/analysis/rrna_intervals
mkdir -p ${OUTDIR}

#grep rRNA from gtf file
grep 'rRNA' ${GTF} > ${OUTDIR}/rRNA_transcripts.gtf

#Create conda enviornments for the following tools
#/globalhome/hxo752/HPC/anaconda3/condabin/conda create -n gtfToGenePred -c bioconda ucsc-gtftogenepred
#/globalhome/hxo752/HPC/anaconda3/condabin/conda create -n genepredtobed -c bioconda ucsc-genepredtobed

#convert the GTF file to genePred format
/globalhome/hxo752/HPC/anaconda3/envs/gtfToGenePred/bin/gtfToGenePred ${OUTDIR}/rRNA_transcripts.gtf ${OUTDIR}/rRNA_transcripts.genePred

#create interval bed file
/globalhome/hxo752/HPC/anaconda3/envs/genepredtobed/bin/genePredToBed ${OUTDIR}/rRNA_transcripts.genePred ${OUTDIR}/rRNA_intervals.bed

#removing overlapping intervals
bedtools merge -i ${OUTDIR}/rRNA_intervals.bed > ${OUTDIR}/rRNA_intervals_merged.bed
