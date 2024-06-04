#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64000
#SBATCH --time=12:00:00
#SBATCH --job-name=picrust_analysis
#SBATCH --partition=batch,guest
#SBATCH --error=/work/claytonlab/ereisher/philzoo2/qiime2/errpi.out
#SBATCH --output=/work/claytonlab/ereisher/philzoo2/qiime2/stdpi.out
#SBATCH --open-mode=append


cd /work/claytonlab/ereisher/philzoo2/qiime2/picrust


module load picrust2/2.4

picrust2_pipeline.py -s /work/claytonlab/ereisher/philzoo2/qiime2/exports/rep-seqs/dna-sequences.fasta -i /work/claytonlab/ereisher/philzoo2/qiime2/exports/table/feature-table-w-taxa.biom -o results
