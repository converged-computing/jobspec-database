#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=CONCOCT
#SBATCH --output=/!!WORKDIR!!/errorOut/concoct_out.o  # < CHANGE
#SBATCH --error=/!!WORKDIR!!/errorOut/concoct_error.e # < CHANGE
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=user@unibe.ch                      # < CHANGE
#-------------------------------------------------------------
# STEP 4 - TEST - CONCOCT 
# tools: concoct 1.1.0
# image source: https://hub.docker.com/r/nanozoo/concoct
# software source: https://github.com/BinPro/CONCOCT
#-------------------------------------------------------------
## !! BIT TO CHANGE !! 
workdir=/path/to/workdir                              # < CHANGE
##-------------------------------------------------------

# 0 --- setup 
datasets_array=($(<datasets.txt))
concoct_sif="${workdir}singularity/concoct.sif"

# loop through datasets 
for dataset in "${datasets_array[@]}"; do

# ---set variables ---
outdir="${workdir}results/04-CONCOCT/${dataset}_bins"
contigs="${workdir}data/assemblies/${dataset}.fa.gz"
bamfile="${workdir}results/01-indexing/${dataset}_indices/${dataset}_aligned.sorted.bam"
mkdir -p ${outdir}
cd ${outdir}


# 1 --- CONCOCT ---
# i - cut up contigs into smaller parts and generate .bed file 
singularity exec \
--bind ${outdir} \
--bind ${bamfile} \
--bind ${contigs} \
--bind ${workdir} \
${concoct_sif} cut_up_fasta.py <( zcat ${contigs}) -c 10000 -o 0 --merge_last -b contigs_10k.bed > contigs_10k.fa
# -c 10000 = chunk size 
# -o 0 = overlap size 
# -b = bedfile to be created 


# ii - generate coverage table 
singularity exec \
--bind ${outdir} \
--bind ${bamfile} \
--bind ${contigs} \
--bind ${workdir} \
${concoct_sif} concoct_coverage_table.py contigs_10k.bed ${bamfile} > coverage_table.tsv


# iii - run concoct 
singularity exec \
--bind ${outdir} \
--bind ${bamfile} \
--bind ${contigs} \
--bind ${workdir} \
${concoct_sif} concoct -t 8 --composition_file contigs_10k.fa --coverage_file coverage_table.tsv -b concoct_output/

# iv- Merge subcontig clustering into original contig clustering
singularity exec \
--bind ${outdir} \
--bind ${bamfile} \
--bind ${contigs} \
--bind ${workdir} \
${concoct_sif} merge_cutup_clustering.py concoct_output/clustering_gt1000.csv > concoct_output/clustering_merged.csv

# v - Extract bins as individual fasta 
mkdir ${outdir}/concoct_output/fasta_bins
singularity exec \
--bind ${outdir} \
--bind ${bamfile} \
--bind ${contigs} \
--bind ${workdir} \
${concoct_sif} extract_fasta_bins.py <(zcat ${contigs}) concoct_output/clustering_merged.csv --output_path concoct_output/fasta_bins
done
