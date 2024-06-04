#!/bin/bash
#SBATCH --job-name=4PopSplit
#SBATCH --output=logs/4PopSplit.out
#SBATCH --error=logs/4PopSplit.err
#SBATCH --time=05:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --mem-per-cpu=2000

module load python/cpython-3.7.0
module load R
module load htslib/1.4.1
module load samtools
module load bcftools

echo "SLURM_JOBID="$SLURM_JOBID
cat snakefile
snakemake --cores all --rerun-incomplete   
#snakemake -j1 --rerun-incomplete 
