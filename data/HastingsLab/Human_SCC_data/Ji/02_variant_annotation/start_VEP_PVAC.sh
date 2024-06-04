#!/bin/bash
#SBATCH --job-name=Merge  # Job name
#SBATCH --mail-type=ALL           # notifications for job done & fail
#SBATCH --mail-user=eknodel@asu.edu # send-to address
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 0-10:00:00
#SBATCH --mem=40000
#SBATCH -q public
#SBATCH -p general

source activate vep_env

module load bcftools-1.14-gcc-11.2.0

snakemake --snakefile VEP_PVACseq.snakefile -j 71 --keep-target-files --rerun-incomplete --cluster "sbatch -q public -p general -n 1 -c 1 --mem=50000 -t 0-10:00:00"

