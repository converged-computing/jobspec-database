#! /bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=170G
#SBATCH --time=72:00:00
#SBATCH --partition=standard
#SBATCH --account=jcbnunez

# This script is a pipeline which applies GATK variant recalibration to VCFs.

#Load Modules
module load gatk
module load vcftools
module load tabix
module load gcc/9.2.0
module load bedtools

#Name of pipeline
PIPELINE=RecalibrateSNPs

#Working folder is core folder where this pipeline is being run.
WORKING_FOLDER=/scratch/yey2sn/phasing_droso_data

#Reference genome
REFERENCE=/project/berglandlab/Dmel_fasta_refences/holo_dmel_6.12.fa

#Intervals to analyze
intervals=/scratch/yey2sn/phasing_droso_data/Intervals_Dmel.txt

#DGRP true SNP panels
# this file is generating by randomly sampling SNPs from the DGRP2 VCF panel.
# http://dgrp2.gnets.ncsu.edu/data.html
# Notice that Prior to sampling, the DGRP2 VCF was liftonver to Dmel6
DGRP=/project/berglandlab/DGRP_freeze2_vcf/dgrp2.v6lift.regexFix.noIndelRegions.noRep.randomSubset.vcf

###########################################################################
#Parameters
#Java
JAVAMEM=160G
CPU=$SLURM_CPUS_ON_NODE
echo "using #CPUs ==" $SLURM_CPUS_ON_NODE
###########################################################################

###########################################################################
###########################################################################
# Begin Pipeline
###########################################################################
###########################################################################
#This part of the pipeline will generate log files to record warnings and completion status

# Move to working directory
cd $WORKING_FOLDER

###########################################################################
###########################################################################
# Make Folder
###########################################################################
###########################################################################

if [[ -d "R_plots_Recalibrate" ]]
then
	echo "Working R_plots_Recalibrate folder exist"
	echo "lets move on"
	date
else 
	echo "folder doesnt exist. lets fix that"
	mkdir $WORKING_FOLDER/R_plots_Recalibrate
	date
fi

if [[ -d "RECALL" ]]
then
	echo "Working RECALL folder exist"
	echo "lets move on"
	date
else 
	echo "folder doesnt exist. lets fix that"
	mkdir $WORKING_FOLDER/RECALL
	date
fi

###########################################################################
###########################################################################
# Generate Folders and files
###########################################################################
###########################################################################

#Interval to analyze
i=`sed -n ${SLURM_ARRAY_TASK_ID}p $intervals`

echo ${i} "is being processed" $(date)

###########################################################################
###########################################################################
# Filter VCFs to retain just SNPs
###########################################################################
###########################################################################

	vcftools \
		--gzvcf $WORKING_FOLDER/${i}.genotyped.raw.vcf.gz \
		--recode \
		--recode-INFO-all \
		--remove-indels \
		--out ${i}.genotypedSNPs.raw
		
#bgzip and tabix
bgzip ${i}.genotypedSNPs.raw.recode.vcf
tabix ${i}.genotypedSNPs.raw.recode.vcf.gz


###########################################################################
###########################################################################
# Random Sample SNPs for validation
###########################################################################
###########################################################################

 bedtools intersect -header \
	-a $WORKING_FOLDER/${i}.genotypedSNPs.raw.recode.vcf.gz \
	-b $DGRP  \
	> ${i}.trueSNPs.vcf

#bgzip and tabix
	bgzip $WORKING_FOLDER/${i}.trueSNPs.vcf
	tabix $WORKING_FOLDER/${i}.trueSNPs.vcf.gz

###########################################################################
###########################################################################
# Run Recalibration
###########################################################################
###########################################################################

#Load R specific dependencies
module load intel/18.0 intelmpi/18.0
module load goolf/7.1.0_3.1.4
module load gdal proj R/4.0.0


  gatk --java-options "-Xmx${JAVAMEM} -Xms${JAVAMEM}" VariantRecalibrator \
      -R $REFERENCE \
      -V $WORKING_FOLDER/${i}.genotypedSNPs.raw.recode.vcf.gz \
	  --resource:dgrp,known=false,training=true,truth=true,prior=15.0 $WORKING_FOLDER/${i}.trueSNPs.vcf.gz \
      --max-gaussians 4 \
      -an DP \
      -an QD \
      -an FS \
      -an SOR \
      -an MQ \
      -an MQRankSum \
      -an ReadPosRankSum \
      -mode SNP \
      -tranche 100.0 -tranche 99.9 -tranche 99.0 -tranche 90.0 \
	  -O $WORKING_FOLDER/RECALL/${i}.recallfile.recal \
	  --tranches-file $WORKING_FOLDER/RECALL/${i}.recalibrate_SNP.tranches \
	  --rscript-file $WORKING_FOLDER/R_plots_Recalibrate/${i}.recalibrate_SNP_plots_merged.R

###########################################################################
###########################################################################
# Run Recalibration
###########################################################################
###########################################################################

gatk --java-options "-Xmx${JAVAMEM} -Xms${JAVAMEM}" ApplyVQSR \
   -R $REFERENCE \
   -V $WORKING_FOLDER/${i}.genotypedSNPs.raw.recode.vcf.gz \
   -O $WORKING_FOLDER/${i}.recalibratedSNP.vcf \
   --truth-sensitivity-filter-level 99.0 \
   --tranches-file $WORKING_FOLDER/RECALL/${i}.recalibrate_SNP.tranches \
   --recal-file $WORKING_FOLDER/RECALL/${i}.recallfile.recal \
   -mode SNP

#bgzip and tabix
	bgzip $WORKING_FOLDER/${i}.recalibratedSNP.vcf
	tabix $WORKING_FOLDER/${i}.recalibratedSNP.vcf.gz

###########################################################################
###########################################################################
# Clean Up
###########################################################################
###########################################################################

rm ${i}.genotypedSNPs.raw.log
rm ${i}.genotypedSNPs.raw.recode.vcf.gz
rm ${i}.genotypedSNPs.raw.recode.vcf.gz.tbi
rm ${i}.trueSNPs.vcf.gz.tbi
rm ${i}.trueSNPs.vcf.gz
rm ${i}.genotypedSNPs.raw.log


echo ${i} "done" $(date)