#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J 16S_NF
#SBATCH -o 16S_NF.%J.out
#SBATCH -e 16S_NF.%J.err
#SBATCH --time=02:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=12

module load nextflow
module load singularity

nextflow run 16SProcessing.nf --in_dir directory/with/fastq/files -profile singularity
