#!/bin/bash

#SBATCH --mail-type=end
#SBATCH --mail-user=zly18810602991@163.com
#SBATCH --job-name=test
#SBATCH --partition=64c512g
#SBATCH -N 1
#SBATCH -J Varscan2
#SBATCH --ntasks-per-node=16
#SBATCH --output=%j.o
#SBATCH --error=%j.e

# ***********************************
# Update: 2023.5.10
# Note: Marked the somaticFilter part and test 
#	Changed the tumor purity to 0.01 for ctDNA
#       Dreased the mutation reads count from 5 to 3
# ************************************


# Usage: sbatch somatic_snv_Varscan2.slurm /lustre/home/acct-medkwf/medkwf4/results/MRD/CC_data/SRR3862160_CC-H007C/bam_19/SRR3862160_CC-H007C.sort.bam /lustre/home/acct-medkwf/medkwf4/results/MRD/CC_data/SRR3862162_CC-H007N/bam_19/SRR3862162_CC-H007N.sort.bam 

# Usage: sbatch somatic_snv_Varscan2.slurm /lustre/home/acct-medkwf/medkwf4/results/MRD/CC_data/SRR3862151_CC-H016C/bam_19/SRR3862151_CC-H016C.sort.bam /lustre/home/acct-medkwf/medkwf4/results/MRD/CC_data/SRR3862147_CC-H016N/bam_19/SRR3862147_CC-H016N.sort.bam

# H009: sbatch somatic_snv_Varscan2.slurm /lustre/home/acct-medkwf/medkwf4/results/MRD/CC_data/SRR3862156_CC-H009N/bam_19/SRR3862156_CC-H009N.sort.bam /lustre/home/acct-medkwf/medkwf4/results/MRD/CC_data/SRR3862157_CC-H009C/bam_19/SRR3862157_CC-H009C.sort.bam

# Initialize a variable with an intuitive name to store the name of the input fastq file
tumor=$1
normal=$2
#project_dir=$3

# Set working dir and annotation dir
#working_dir=${project_dir}/../Varscan2
#mkdir -p ${working_dir}
#annovar=${project_dir}/../ANNOVAR
#mkdir -p ${annovar}
#bed=${project_dir}/../bed
#mkdir -p ${bed}
#BRC=${project_dir}/../bam-readcount

# Get base name
baseSuper=`basename ${tumor} .sort.bam`
base=$(echo ${baseSuper} | cut -d'_' -f 2)  # get the filename with basename function
echo "base is ${base}"
echo "Tumor sample is ${tumor}"
echo "Normal sample is ${normal}"

# Set working dir and other result dirs
project_dir=/dssg/home/acct-medkwf/medkwf4/results/ctDNA_EQA/${base}
mkdir -p ${project_dir}
working_dir=${project_dir}/Varscan2
mkdir -p ${working_dir}
annovar=${project_dir}/ANNOVAR
mkdir -p ${annovar}
bed=${project_dir}/bed
mkdir -p ${bed}
BRC=${project_dir}/bam-readcount
mkdir -p ${BRC}

# Directory with reference genome
genome_19=/dssg/home/acct-medkwf/medkwf4/database/GATK/hg19/ucsc.hg19.fasta

# Set up environment
module load miniconda3
source activate dna

# specify the number of cores to use
cores=4

# Set output files
normal_tumor_mpileup=${working_dir}/${base}.normal-tumor.mpileup
out_snv=${working_dir}/${base}.somatic.vcf
out_snv_merge=${working_dir}/${base}.snv.hc
filter_snv=${working_dir}/${base}.ps.hc.sf.vcf
target_bed=${bed}/${base}.target.bed
BRC_readcount=${BRC}/${base}.readcount
output_Varscan=${working_dir}/${base}.fianl.vcf
#output_Varscan=${working_dir}/${base}.somatic.vcf.snp.Somatic.hc
output_Varscan_failed=${working_dir}/${base}.final.fail.vcf 
annovar_aviinput_Varscan=${annovar}/${base}.Varscan
annovar_annotated_Varscan=${annovar}/${base}.Varscan.csv

# BED file
# 602panel:
region_bed=/dssg/home/acct-medkwf/medkwf4/reference/bed/602panel/NanOnco_Plus_Panel_v2.0_Covered_b37_cg.parY2X.sort.chr.bed

# Use Varscan2
# pieup firstly (necessary for Varscan2)
# -q, --min-MQ INT Minimum mapping quality for an alignment to be used [0]
# -C, --adjust-MQ INT
# A, --count-orphans Do not skip anomalous read pairs in variant calling. Anomalous read pairs are those marked in the FLAG field as paired in sequencing but without the properly-paired flag set.
#-l, --positions FILE    skip unlisted positions (chr pos) or regions (BED) 

#samtools mpileup -f ${genome_19} -B -q 10 -Q 10 -l ${region_bed} -x ${normal} ${tumor}> ${normal_tumor_mpileup}


#java -jar ~/software/VarScan.v2.3.9.jar somatic \
#${normal_tumor_mpileup} \
#${out_snv} \
#--mpileup 1 \
#--output-vcf 1 \
#--min-coverage-normal 10 \
#--min-var-freq 0.001 \
#--tumor-purity 0.01 \

# Filter as recommended by official doc https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4278659/
# Region should be specified otherwise the output file is super large and cost much time
#java -jar ~/software/VarScan.v2.3.9.jar processSomatic ${out_snv}.snp --min-tumor-freq 0.001 --max-normal-freq 0.05 --p-value 0.5
#java -jar ~/software/VarScan.v2.3.9.jar processSomatic ${out_snv}.indel --min-tumor-freq 0.001 --max-normal-freq 0.05 --p-value 0.5
#less ${out_snv}.snp.Somatic.hc > ${filter_snv}
#less ${out_snv}.indel.Somatic.hc | grep -vE "^#" >> ${filter_snv}

# Mark this part
#java -jar ~/software/VarScan.v2.3.9.jar somaticFilter ${out_snv_merge} \
#--min-var-freq  0.001 \
#--min-coverage 20 \
#--output-file ${filter_snv}

#grep -v "#" ${filter_snv} | awk '{a=$2+10;print $1"\t"$2"\t"a}' > ${target_bed}
# -w maximum Warning times
#singularity exec /dssg/home/acct-medkwf/medkwf4/software/bam-readcount.sif bam-readcount \
#-q 20 \
#-b 20 \
#-w 100 \
#-f ${genome_19} \
#-l ${target_bed} \
#${tumor} > ${BRC_readcount}

#java -jar ~/software/VarScan.v2.3.9.jar fpfilter \
#${filter_snv} \
#${BRC_readcount} \
#--min-var-count 3 \
#--min-var-basequal 20 \
#--min-ref-basequal 20 \
#--min-var-freq 0.001 \
#--output-file ${output_Varscan} \
#--filtered-file ${output_Varscan_failed}

/dssg/home/acct-medkwf/medkwf4/software/annovar/convert2annovar.pl --format vcf4old --includeinfo --withzyg ${output_Varscan} > ${annovar_aviinput_Varscan}
/dssg/home/acct-medkwf/medkwf4/software/annovar/table_annovar.pl ${annovar_aviinput_Varscan} /dssg/home/acct-medkwf/medkwf4/software/annovar/humandb -buildver hg19 -out ${annovar_annotated_Varscan} -polish -remove -protocol refGene,clinvar_20170905,icgc28,gnomad_exome,EAS.sites.2014_10,avsnp144,EAS.sites.2015_08,cosmic70,nci60  -operation g,f,f,f,f,f,f,f,f --otherinfo -csvout

