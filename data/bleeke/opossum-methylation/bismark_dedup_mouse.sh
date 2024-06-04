#!/bin/bash
# BSseq mouse allele-specific
#SBATCH --job-name=dedup
#SBATCH --ntasks=8
#SBATCH --time=3-00:00:0
#SBATCH --mem=60G
#SBATCH --partition=cpu
#SBATCH --array=1-36
#SBATCH --output=bismark_dedup_mouse_%A_%a.out




# BJL 
# 20190902
# for deduplication of BS-seq bams 
# from mouse allele-specific analysis
# data has been Bismark mapped single end non-dir against n-masked genome
# and run through SNPsplit programme


echo "begin"
date

# variables, files and directories

BAMDIR=/camp/lab/turnerj/working/Bryony/mouse_adult_xci/allele_specific/data/bs-seq/bams/split_bams

INFILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" 20190902_bams_to_dedup.txt)


# load bismark

ml purge
ml Bismark
echo "modules loaded are:" 
ml




## DEDUPLICATE BAMS ##

cd $BAMDIR

deduplicate_bismark --bam $INFILE


echo "end"
date

