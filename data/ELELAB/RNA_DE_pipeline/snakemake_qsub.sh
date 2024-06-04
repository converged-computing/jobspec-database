#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=dtu_00011 -A dtu_00011
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N 161_rosetta_rna_GENCODE
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -e rnaseq_test.err
#PBS -o rnaseq_test.log
### Only send mail when job is aborted or terminates abnormally
#PBS -m n
### Number of nodes
#PBS -l nodes=1:ppn=38:fatnode
### Memory
#PBS -l mem=350gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 12 hours)
#PBS -l walltime=180:00:00

# Go to the directory from where the job was submitted (initial directory is $HOME)
MY_dir=/home/projects/dtu_00011/data/icope_analysis/others/rna/rnaseq/gene_expression/runs/rosetta_161_samples/code/RNA_DE_pipeline
echo Working directory is $MY_dir
cd $MY_dir

### Here follows the user commands:
# Define number of processors
NPROCS=38
echo This job has allocated $NPROCS cores

# Load all required modules for the job
module load tools
module load miniconda3/4.12.0
conda init --all
conda activate /home/people/niktom/.conda/envs/snakemake_6.13.1/


# This is where the work is done
# Make sure that this script is not bigger than 64kb ~ 150 lines, otherwise put in seperat script and execute from here
#export SNAKEMAKE_OUTPUT_CACHE=/home/people/niktom/nik_projects/snakemake/test_RNAseq/rna-seq-star-deseq2/resources
#snakemake --cores 30 --use-conda --cache get_genome get_annotation genome_faidx bwa_index star_index star_genome
#Conda environment is sourced from local conda repository
snakemake --cores 38 --use-conda --rerun-incomplete
