#!/bin/bash
# SNPsplit
#SBATCH --job-name=SNPsplit
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem=40G
#SBATCH --partition=cpu
#SBATCH --array=1,6-8
#SBATCH --output=SNPsplit_%A_%a.out


####------------------------------------------------------------------####
## script for running SNPsplit on BS-seq data
## the data was mapped with Bismark
####------------------------------------------------------------------####


echo "IT HAS BEGUN"
date



## create variable containing library number
LIBNUM=${SLURM_ARRAY_TASK_ID}
echo "we are working on library number" "$LIBNUM"



# directories 

BAMDIR=/camp/lab/turnerj/working/Bryony/opossum_adult/allele-specific/data/bs-seq/bams # where the mapped files to be split are

SNPDIR=/camp/lab/turnerj/working/shared_projects/OOPs/genome # where the list of SNPs is - using v2 with correct order of parental snps

OUTDIR=/camp/lab/turnerj/working/Bryony/opossum_adult/allele-specific/data/bs-seq/bams/split_bams # where to put the SNPsplit output files




# run SNPsplit

ml purge
ml Anaconda2
source activate SNPsplit
echo "loaded modules are:"
ml

echo "running SNPsplit"

SNPsplit --bisulfite --conflicting -o $OUTDIR --snp_file $SNPDIR/clean.snps.jvdb.v2.txt  $BAMDIR/LEE62A${LIBNUM}_*bt2.bam



echo "IT HAS FINISHED"
date