#!/bin/sh

#PBS -l nodes=1:ppn=48
#PBS -l mem=400g
#PBS -l walltime=1:00:00
#PBS -N idxs
#PBS -q cgsd 
#PBS -e .idxe
#PBS -o .idxo
#qsub snakemake_viral_calling/scripts/merge_idxstats.sh


# module purge
module add miniconda3/4.12.0
module add parallel
source activate mypy3

# chmod +x snakemake_viral_calling/scripts/merge_idxstats.py
# python snakemake_viral_calling/scripts/merge_idxstats.py

# chmod +x snakemake_viral_calling/scripts/merge_idx3.py
# python snakemake_viral_calling/scripts/merge_idx3.py

# chmod +x snakemake_viral_calling/scripts/vir_top10.py
# python snakemake_viral_calling/scripts/vir_top10.py


chmod +x snakemake_viral_calling/scripts/sumctg2ref.py

python snakemake_viral_calling/scripts/sumctg2ref.py
