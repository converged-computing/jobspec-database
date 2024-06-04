#!/bin/bash
#PBS -N meryl 
#PBS -P xl04
#PBS -q express
#PBS -l walltime=1:00:00
#PBS -l mem=4GB
#PBS -l ncpus=2
#PBS -l storage=gdata/xl04+gdata/if89
#PBS -l wd
#PBS -o /g/data/xl04/ka6418/github/ausarg/nextflow/non-experimental
#PBS -M kirat.alreja@anu.edu.au

module load nextflow
nextflow run /g/data/xl04/ka6418/github/ausarg/nextflow/non-experimental/genejigsaw.nf -config /g/data/xl04/ka6418/github/ausarg/nextflow/non-experimental/genejigsaw.config --topfolder /g/data/xl04/ka6418/github/ausarg/nextflow/sql_integration