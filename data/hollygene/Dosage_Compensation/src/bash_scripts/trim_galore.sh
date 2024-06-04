#PBS -S /bin/bash
#PBS -N j_TrimGalore
#PBS -q batch
#PBS -l nodes=1:ppn=1:AMD
#PBS -l walltime=480:00:00
#PBS -l mem=10gb

cd $PBS_O_WORKDIR

module load Trim_Galore/0.4.5-foss-2016b
#sample 3C
trim_galore /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/fastq/Holly_3C_S5_R1_001.fastq --fastqc --phred33 -q 20  --illumina

trim_galore /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/fastq/Holly_3C_S5_R1_001.fastq --fastqc --phred33 -q 20  -a GCTAATTCTCGTGTACAACACTTACAACAGAGACAAAGAGTTTTTTCTGG
#21
trim_galore /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/fastq/Holly-GC-21-redo_S8_R1_001.fastq --fastqc --phred33 -q 20 -a GATCGGAAGAGCACACGTCTGAACTCCAGTCACCTGAAGCTATCTCGTAT

#61
trim_galore /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/fastq/Holly-GC-61-redo_S11_R1_001.fastq --fastqc --phred33 -q 20 -a GATCGGAAGAGCACACGTCTGAACTCCAGTCACCGCTCATTATCTCGTAT

#66
trim_galore /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/fastq/Holly-66B-redo-136232_S14_R1_001.fastq --fastqc --phred33 -q 20 -a GATCGGAAGAGCACACGTCTGAACTCCAGTCACGAATTCGTATCTCGTAT


trim_galore /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/fastq/Holly_69C_S22_R1_001.fastq --fastqc --phred33 -q 20 -a GATCGGAAGAGCACACGTCTGAACTCCAGTCACCTGAAGCTATCTCGTAT
