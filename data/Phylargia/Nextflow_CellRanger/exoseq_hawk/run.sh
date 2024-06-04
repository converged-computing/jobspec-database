#!/bin/bash
#SBATCH -p compute
#SBATCH --mem=20GB
#SBATCH --ntasks=1
#SBATCH -t 24:00:00
#SBATCH -o slurm/%J.out
#SBATCH -e slurm/%J.err
#SBATCH --job-name=exoseq_hawk
#SBATCH --account=scw1557 # Change Project group

module load nextflow/21.10.6
cd /scratch/c.c1845715/nextflow_cellranger/exoseq_hawk # Change User ID

nextflow run main.nf --genome mouse --input 'input/input.csv' -with-trace
