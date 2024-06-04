#!/bin/bash
#SBATCH -J fastp
#SBATCH -o fastp_%A_%a.out
#SBATCH -e fastp_%A_%a.err
#SBATCH --array=1-100
#SBATCH -n 8 -N 1
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yelena.pacheco@usda.gov

#trimm adaptors from raw read fastq files 

#create output directories for trimmed reads and output files 
if [ ! -d trimmed_reads ]
then
   mkdir -p trimmed_reads
fi
 
 
 
if [ ! -d fastp_trimming_docs ]
then
  mkdir -p fastp_trimming_docs
fi
 
 

module load miniconda
source activate /project/leafbeetle_phylogenomics/yelena/hybpiper

#input.list is a list of all the *_1.fastq files in the directory
infile=$(cat input.list | head -n ${SLURM_ARRAY_TASK_ID} | tail -n1)

base=$(basename ${infile} _1.fastq)

fastp -w 4 --dont_overwrite --disable_quality_filtering --correction --length_required 21 \
--in1 ${base}_1.fastq --in2 ${base}_2.fastq \
--out1 trimmed_reads/${base}_P_R1.fastq --out2 trimmed_reads/${base}_P_R2.fastq \
--unpaired1 trimmed_reads/${base}_U.fastq --unpaired2 trimmed_reads/${base}_U.fastq \
--failed_out fastp_trimming_docs/${base}_failed.fastq \
-j fastp_trimming_docs/${base}.fastp_report.json \
-h fastp_trimming_docs/${base}.fastp_report.html

conda deactivate
