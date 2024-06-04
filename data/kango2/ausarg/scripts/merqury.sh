#!/bin/bash
#PBS -N merqury_bash
#PBS -P xl04
#PBS -q normal
#PBS -l walltime=48:00:00
#PBS -l mem=2GB
#PBS -l ncpus=1
#PBS -l storage=gdata/xl04+gdata/if89
#PBS -l wd
#PBS -M kirat.alreja@anu.edu.au

module load nextflow 
nextflow run /g/data/xl04/ka6418/github/ausarg/scripts/merqury.nf --fileList ${files} --sample ${sample} --tech ${tech} --kmer ${kmer} --output ${output} --fasta ${ref}