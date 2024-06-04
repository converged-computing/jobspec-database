#!/bin/bash
#SBATCH --chdir=./
#SBATCH --job-name=bam
#SBATCH --partition quanah
#SBATCH --nodes=1 --ntasks=19
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-38
#SBATCH --mail-user=arrice@ttu.edu
#SBATCH --mail-type=ALL

module load intel java bwa samtools singularity

export SINGULARITY_CACHEDIR="/lustre/work/arrice/singularity-cachedir"

# define main working directory
workdir=/lustre/scratch/arrice/Ch1_Leiothlypis

basename_array=$( head -n${SLURM_ARRAY_TASK_ID} ${workdir}/basenames.txt | tail -n1 )

# define the reference genome
refgenome=${workdir}/00_ref_genome/ncbi_dataset/data/GCA_009764595.1/GCA_009764595.1_bGeoTri1.pri_genomic.fna

# run bbduk
/lustre/work/jmanthey/bbmap/bbduk.sh in1=${workdir}/00_fastq/${basename_array}_R1.fastq.gz in2=${workdir}/00_fastq/${basename_array}_R2.fastq.gz out1=${workdir}/01_cleaned/${basename_array}_R1.fastq.gz out2=${workdir}/01_cleaned/${basename_array}_R2.fastq.gz minlen=50 ftl=10 qtrim=rl trimq=10 ktrim=r k=25 mink=7 ref=/lustre/work/jmanthey/bbmap/resources/adapters.fa hdist=1 tbo tpe
# "DUK" stands for decontamination using kmers. It combines a bunch of tools for quality trimming, adapter trimming, filtering, etc. 
# "in1" and "in2"= Input files. "out1" and "out2"= Output files. 
# "minlen=50" throws out reads shorter than 50 bp. 
# "ftl=10" trims the leftmost 10 bases. 
# "qtrim=rl" Quality-trims the right and left side of the reads. 
# "trimq=10" Quality-trims reads with Phred scores under 10. 
# "ktrim=r" Trims out reference kmers and all bases to the right of it after it's matched to a read. This is the normal mode for adapter trimming.
# "k=25" Something to do with kmers. 
# "mink=7" Looks for kmers between 7 and 25 base pairs. 
# "ref=/lustre/work/jmanthey/bbmap/resources/adapters.fa" The reference file that has all the adapter kmers that you're filtering out. 
# "hdist=1" Hamming distance! (whatever that means)
# "tbo" Not sure what this does.  
# "tpe" Trims both reads to the same length in case a kmer was only detected in one of them. 

# define the location of the reference mitogenomes
mito=${workdir}/01b_mtDNA/songbird_pero_mitogenomes.fasta

# run bbsplit
/lustre/work/jmanthey/bbmap/bbsplit.sh in1=${workdir}/01_cleaned/${basename_array}_R1.fastq.gz in2=${workdir}/01_cleaned/${basename_array}_R2.fastq.gz ref=${mito} basename=${workdir}/01b_mtDNA/${basename_array}_%.fastq.gz outu1=${workdir}/01b_mtDNA/${basename_array}_R1.fastq.gz outu2=${workdir}/01b_mtDNA/${basename_array}_R2.fastq.gz

# remove unnecessary bbsplit output files
rm ${workdir}/01b_mtDNA/${basename_array}_R1.fastq.gz
rm ${workdir}/01b_mtDNA/${basename_array}_R2.fastq.gz

bwa mem -t 12 ${refgenome} ${workdir}/01_cleaned/${basename_array}_R1.fastq.gz ${workdir}/01_cleaned/${basename_array}_R2.fastq.gz > ${workdir}/01_bam_files/${basename_array}.sam
# This maps sequences to the reference genome. 
# -t denotes number of threads. 
# Creates a .sam file. 

# convert sam to bam
samtools view -b -S -o ${workdir}/01_bam_files/${basename_array}.bam ${workdir}/01_bam_files/${basename_array}.sam
# Sam files are massive text files depicting alignments. Bam files are their binary equivalent, and they take up significantly less space. 

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
# "--RGLB" stands for Read-Group library. 
# "--RGPL" stands for Read-Group platform, which is illumina in this case. 
# "--RGPU" stands for Read-Group platform unit.
# "--RGSM" stands for Read-Group sample name. 

# remove cleaned and sorted bam file
rm ${workdir}/01_bam_files/${basename_array}_cleaned_sorted.bam

# remove duplicates to sorted, cleaned, and read grouped bam file (creates final bam file)
singularity exec $SINGULARITY_CACHEDIR/gatk_4.2.3.0.sif gatk MarkDuplicates --REMOVE_DUPLICATES true --MAX_FILE_HANDLES_FOR_READ_ENDS_MAP 100 -M ${workdir}/01_bam_files/${basename_array}_markdups_metric_file.txt -I ${workdir}/01_bam_files/${basename_array}_cleaned_sorted_rg.bam -O ${workdir}/01_bam_files/${basename_array}_final.bam

# remove sorted, cleaned, and read grouped bam file
rm ${workdir}/01_bam_files/${basename_array}_cleaned_sorted_rg.bam

# index the final bam file
samtools index ${workdir}/01_bam_files/${basename_array}_final.bam
