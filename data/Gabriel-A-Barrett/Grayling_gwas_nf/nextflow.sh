#!/usr/bin/sh
#SBATCH --job-name=nextflow
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --mem=5G
module load nextflow/22.04.0

nextflow run main.nf -entry NF_GWAS -resume -with-singularity gwas-nf.sif

