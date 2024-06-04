#!/bin/bash

#SBATCH --job-name=run_pileup_founders
#SBATCH --partition=compute  # ==queue
#SBATCH --nodes=1            # number of nodes
#SBATCH --ntasks=1           # number of cores
#SBATCH --mem=60GB           # memory pool for all cores
#SBATCH --time=32:00:00      # time (HH:MM:SS)
#SBATCH --output=%x.o%A_%a   # stdout and stderr
#SBATCH --array=1

module load singularity

REPO_BASE=$(pwd)
BAM=supporting_files/toy_10X/toy_reads.final.bam
VCF=$REPO_BASE/variants/CC_founders_v4.snps.vcf.gz
BC=supporting_files/toy_10X/toy_barcode.txt
OUTDIR=$REPO_BASE/demuxlet_results/toy
singularity run $REPO_BASE/containers/popscle-1.0.sif dsc-pileup \
    --sam $BAM --vcf <(zcat $VCF) --out $OUTDIR \
    --sam-verbose 10000000 --vcf-verbose 250000 \
    --tag-group CB --tag-UMI UB --group-list $BC
