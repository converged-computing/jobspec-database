#!/bin/bash
#SBATCH --chdir=./
#SBATCH --job-name=setophaga_bam
#SBATCH --partition quanah
#SBATCH --nodes=1 --ntasks=12
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-54

module load intel java bwa samtools singularity 

export SINGULARITY_CACHEDIR="/lustre/work/johruska/singularity-cachedir"

# define main working directory
workdir=/lustre/scratch/johruska/setophaga

basename_array=$( head -n${SLURM_ARRAY_TASK_ID} ${workdir}/setophaga_basenames.txt | tail -n1 )

# define the reference genome
refgenome=/lustre/scratch/johruska/central_america_pine_oak/parulidae_ref/GCA_001746935.2_mywa_2.1_genomic.fna

# define the location of the reference mitogenomes
mito=/lustre/scratch/johruska/central_america_pine_oak/parulidae_ref/C_canadensis_mito_NC_041111.1.fasta

# run bbduk
/lustre/work/jmanthey/bbmap/bbduk.sh in1=${workdir}/00_fastq/${basename_array}_R1.fastq.gz in2=${workdir}/00_fastq/${basename_array}_R2.fastq.gz out1=${workdir}/01_cleaned/${basename_array}_R1.fastq.gz out2=${workdir}/01_cleaned/${basename_array}_R2.fastq.gz minlen=50 ftl=10 qtrim=rl trimq=10 ktrim=r k=25 mink=7 ref=/lustre/work/jmanthey/bbmap/resources/adapters.fa hdist=1 tbo tpe

# run bbsplit
/lustre/work/jmanthey/bbmap/bbsplit.sh in1=${workdir}/01_cleaned/${basename_array}_R1.fastq.gz in2=${workdir}/01_cleaned/${basename_array}_R2.fastq.gz ref=${mito} basename=${workdir}/01_mtDNA/${basename_array}_%.fastq.gz outu1=${workdir}/01_mtDNA/${basename_array}_R1.fastq.gz outu2=${workdir}/01_mtDNA/${basename_array}_R2.fastq.gz

# remove unnecessary bbsplit output files
rm ${workdir}/01_mtDNA/${basename_array}_R1.fastq.gz
rm ${workdir}/01_mtDNA/${basename_array}_R2.fastq.gz

# run bwa mem
bwa mem -t 12 ${refgenome} ${workdir}/01_cleaned/${basename_array}_R1.fastq.gz ${workdir}/01_cleaned/${basename_array}_R2.fastq.gz > ${workdir}/01_bam_files/${basename_array}.sam

# convert sam to bam
samtools view -b -S -o ${workdir}/01_bam_files/${basename_array}.bam ${workdir}/01_bam_files/${basename_array}.sam

# remove sam
rm ${workdir}/01_bam_files/${basename_array}.sam

# clean up the bam file
singularity exec $SINGULARITY_CACHEDIR/gatk_4.2.3.0.sif gatk CleanSam -I ${workdir}/01_bam_files/${basename_array}.bam -O ${workdir}/01_bam_files/${basename_array}_cleaned.bam

# remove the raw bam
rm ${workdir}/01_bam_files/${basename_array}.bam

# sort the cleaned bam file
singularity exec $SINGULARITY_CACHEDIR/gatk_4.2.3.0.sif gatk SortSam -I ${workdir}/01_bam_files/${basename_array}_cleaned.bam -O ${workdir}/01_bam_files/${basename_array}_cleaned_sorted.bam --SORT_ORDER coordinate

# remove the cleaned bam file
rm ${workdir}/01_bam_files/${basename_array}_cleaned.bam

# add read groups to sorted and cleaned bam file
singularity exec $SINGULARITY_CACHEDIR/gatk_4.2.3.0.sif gatk AddOrReplaceReadGroups -I ${workdir}/01_bam_files/${basename_array}_cleaned_sorted.bam -O ${workdir}/01_bam_files/${basename_array}_cleaned_sorted_rg.bam --RGLB 1 --RGPL illumina --RGPU unit1 --RGSM ${basename_array}

# remove cleaned and sorted bam file
rm ${workdir}/01_bam_files/${basename_array}_cleaned_sorted.bam

# remove duplicates to sorted, cleaned, and read grouped bam file (creates final bam file)
singularity exec $SINGULARITY_CACHEDIR/gatk_4.2.3.0.sif gatk MarkDuplicates --REMOVE_DUPLICATES true --MAX_FILE_HANDLES_FOR_READ_ENDS_MAP 100 -M ${workdir}/01_bam_files/${basename_array}_markdups_metric_file.txt -I ${workdir}/01_bam_files/${basename_array}_cleaned_sorted_rg.bam -O ${workdir}/01_bam_files/${basename_array}_final.bam

# remove sorted, cleaned, and read grouped bam file
rm ${workdir}/01_bam_files/${basename_array}_cleaned_sorted_rg.bam

# index the final bam file
samtools index ${workdir}/01_bam_files/${basename_array}_final.bam
