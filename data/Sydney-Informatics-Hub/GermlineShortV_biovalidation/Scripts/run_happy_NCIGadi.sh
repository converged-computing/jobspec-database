#!/bin/bash

##########################################################################
#
# Platform: NCI Gadi HPC
# Usage: qsub run_happy.pbs (or bash run_happy.sh)
# Version: 1.0
#
# For more details see: https://github.com/Sydney-Informatics-Hub/GermlineShortV_biovalidation
#
# If you use this script towards a publication, please acknowledge the
# Sydney Informatics Hub (or co-authorship, where appropriate).
#
# Suggested acknowledgement:
# The authors acknowledge the support provided by the Sydney Informatics Hub,
# a Core Research Facility of the University of Sydney. This research/project
# was undertaken with the assistance of resources and services from
# the Australian BioCommons which is enabled by NCRIS via Bioplatforms Australia funding.
#
##########################################################################

#PBS -P 
#PBS -N happy_SIF
#PBS -l walltime=03:00:00
#PBS -l ncpus=1
#PBS -l mem=10GB
#PBS -q normal
#PBS -W umask=022
#PBS -l wd
#PBS -e ./Logs/happy.e
#PBS -o ./Logs/happy.o
#PBS -l storage=

# Load singularity
module load singularity
module load htslib/1.12
module load bcftools/1.12

# define variables
vcfdir= #name of directories where vcfs are stored
sample= #name of query vcf
truth= #name of truth vcf
refdir=
ref=${refdir}/ #name of ref.fasta
outdir=
out=${outdir}/${sample}.happy

# make logs directory in current directory
mkdir -p ./Logs

# Pull hap.py container
export SINGULARITY_BINDPATH=${outdir}
SIF=${outdir}/SIF #specify directory to save the singularity image file

mkdir ${SIF}

echo "PREPARING HAP.PY CONTAINER..."
singularity pull --dir ${SIF} docker://quay.io/biocontainers/hap.py:0.3.14--py27h5c5a3ab_0

# remove INFO,FORMAT fields not used by hap.py from query vcf to avoid errors
echo "CLEANING UP QUERY VCF FOR HAPPY..."

bcftools annotate -Oz -x INFO,FORMAT ${vcfdir}/${sample}.vcf.gz -o - | \
        bcftools sort -Oz - -o ${vcfdir}/${sample}_geno_sorted.vcf.gz

# index sorted query vcf
echo "INDEXING QUERY VCF..."

tabix -f ${vcfdir}/${sample}_geno_sorted.vcf.gz

# remove overlapping variants in truth vcf - these throw an error and hap.py fails
echo "PREPARING TRUTH VCF TO AVOID ERRORS..."
bcftools view -e "%FILTER='SiteConflict'" -Oz ${vcfdir}/${truth}.vcf.gz -o ${vcfdir}/${truth}_clean.vcf.gz

# index cleaned truth vcf
tabix ${vcfdir}/${truth}_clean.vcf.gz

# run hap.py using cleaned truth vcf and sample vcf
echo "RUNNING HAP.PY VCF COMPARISON TOOL..."

singularity run ${SIF}/hap.py_0.3.14--py27h5c5a3ab_0.sif hap.py \
	${vcfdir}/${truth}_clean.vcf.gz \
	${vcfdir}/${sample}_geno_sorted.vcf.gz \
	--report-prefix ${out} \
	-r ${ref} \
	-L --preprocess-truth --usefiltered-truth --fixchr \
	--roc QUAL \
	--engine xcmp \
	--verbose
