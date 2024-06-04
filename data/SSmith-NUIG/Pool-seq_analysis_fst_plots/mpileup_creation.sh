#!/bin/sh 
#SBATCH --job-name="gene_pile"
#SBATCH -o /data/ssmith/logs/mpileup_creation_%A_%a.out
#SBATCH -e /data/ssmith/logs/mpileup_creation_%A_%a.err
#SBATCH -p highmem
#SBATCH -N 1
#SBATCH -n 12
#"$SLURM_ARRAY_TASK_ID"

source /home/ssmith/.bashrc
source activate wgs_env

samtools mpileup \
-f /data/ssmith/c_l_genome/apis_c_l_genome.fa -q 20 -B \
-b /data/ssmith/scripts/colony_analysis/jan_analysis/pooled_pop_pipeline/markus_scripts/bam_file_list_smartbees.txt \
-l /data/ssmith/scripts/colony_analysis/jan_analysis/pooled_pop_pipeline/matching_snps_locations.txt \
> /data2/ssmith/matching_snps_pop_cerana.mpileup
