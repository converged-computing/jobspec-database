#!/bin/bash
#SBATCH --job-name=unlock_snakemake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --time=00:02:00

module load tools/miniconda/python3.8/4.9.2
conda activate exonexaminer

snakemake --unlock
