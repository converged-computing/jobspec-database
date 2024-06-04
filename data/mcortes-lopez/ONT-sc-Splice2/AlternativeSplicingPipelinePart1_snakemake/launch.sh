#!/bin/bash

#SBATCH --job-name=test
#SBATCH --mem=20G
#SBATCH --partition=pe2
#SBATCH --output=test.log
#SBATCH --error=test.err
#SBATCH --cpus-per-task=5

#module load anaconda3/10.19
module load java/1.9
module load samtools
module load bedtools
module load racon
#set +eu
#source activate use_mamba
source activate sicelore2.0
#module load snakemake 
#set -eu
which python
snakemake -v
snakemake --snakefile Snakefile_multi \
  --configfile config.yml \
  --stats stats.txt \
  --profile ../AlternativeSplicingPipelinePart1_snakemake/profiles/profile_snakemake/ 
  #--cluster "sbatch --error=logs/test_%j_err.log --output=logs/test_%j_out.log --mem=100G --cpus-per-task=10"
