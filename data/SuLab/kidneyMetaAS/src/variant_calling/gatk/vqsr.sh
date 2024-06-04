#!/bin/bash
#PBS -l nodes=1:ppn=8
#PBS -l mem=31gb
#PBS -l walltime=30:00:00
#PBS -m abe
#PBS -M lhgioia@scripps.edu

module load R/3.1.0
module load gatk/3.3-0

gVCF_dir="./results/variant_calling/gatk"
recal_dir="./results/variant_calling/gatk"

hg19_reference="/UCSC/hg19/Sequence/WholeGenomeFasta/genome.fa"
hapmap="/GATK_resources/hapmap_3.3.hg19.vcf"
omni="/GATK_resources/1000G_omni2.5.hg19.vcf"
KG="/GATK_resources/1000G_phase1.snps.high_confidence.hg19.sites.vcf"
dbsnp="/GATK/hg19/dbsnp_137.hg19.vcf"
mills="/GATK/hg19/Mills_and_1000G_gold_standard.indels.hg19.vcf"

java -Xmx28g -jar `which GenomeAnalysisTK.jar` \
   -T VariantRecalibrator \
   -R $hg19_reference \
   -input $gVCF_dir/final.vcf \
   -nt 8 \
   -resource:hapmap,known=false,training=true,truth=true,prior=15.0 $hapmap \
   -resource:omni,known=false,training=true,truth=true,prior=12.0 $omni \
   -resource:1000G,known=false,training=true,truth=false,prior=10.0 $KG \
   -resource:dbsnp,known=true,training=false,truth=false,prior=2.0 $dbsnp \
   -an QD -an MQ -an MQRankSum -an ReadPosRankSum -an FS -an SOR -an InbreedingCoeff \
   -mode SNP \
   -tranche 100.0 -tranche 99.9 -tranche 99.0 -tranche 90.0 \
   -recalFile $recal_dir/recalibrate_SNP.recal \
   -tranchesFile $recal_dir/recalibrate_SNP.tranches \
   -rscriptFile $recal_dir/recalibrate_SNP_plots.R

java -Xmx28g -jar `which GenomeAnalysisTK.jar` \
   -T ApplyRecalibration \
   -R $hg19_reference \
   -input $gVCF_dir/final.vcf \
   -tranchesFile $recal_dir/recalibrate_SNP.tranches \
   -recalFile $recal_dir/recalibrate_SNP.recal \
   -o $recal_dir/recalibrate_SNP.vcf \
   --ts_filter_level 99.5 \
   -mode SNP

java -Xmx28g -jar `which GenomeAnalysisTK.jar` \
   -T VariantRecalibrator \
   -R $hg19_reference \
   -input $recal_dir/recalibrate_SNP.vcf \
   --maxGaussians 4 \
   -resource:mills,known=false,training=true,truth=true,prior=12.0 $mills \
   -resource:dbsnp,known=true,training=false,truth=false,prior=2.0 $dbsnp \
   -an QD -an FS -an SOR -an ReadPosRankSum -an MQRankSum -an InbreedingCoeff \
   -mode INDEL \
   -tranche 100.0 -tranche 99.9 -tranche 99.0 -tranche 90.0 \
   -recalFile $recal_dir/recalibrate_INDEL.recal \
   -tranchesFile $recal_dir/recalibrate_INDEL.tranches \
   -rscriptFile $recal_dir/recalibrate_INDEL_plots.R

java -Xmx28g -jar `which GenomeAnalysisTK.jar` \
   -T ApplyRecalibration \
   -R $hg19_reference \
   -input $recal_dir/recalibrate_SNP.vcf \
   -tranchesFile $recal_dir/recalibrate_INDEL.tranches \
   -recalFile $recal_dir/recalibrate_INDEL.recal \
   -o $recal_dir/recal_SNP_INDEL.vcf \
   --ts_filter_level 99.0 \
   -mode INDEL

exit 0
