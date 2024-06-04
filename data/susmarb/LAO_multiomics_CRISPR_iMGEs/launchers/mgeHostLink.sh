#!/bin/bash -l
#OAR -n mgeHostLink_test
#OAR -l nodes=1/core=1,walltime=4

source /home/users/smartinezarbas/git/gitlab/CRISPR_MGE_pipeline/src/preload_modules.sh

export THREADS='1'

export TS_DIR='/work/users/smartinezarbas/comparative_analysis/AmazonRiver/IMP_results'
export TS_SAMPLES='TS2 TD2'
export DB_FA_DIR='/work/users/smartinezarbas/comparative_analysis/AmazonRiver/IMP_results/Assemblies'
# BIN_DICT should be generated after the dereplication of the bins
export BIN_DICT='/work/users/smartinezarbas/comparative_analysis/AmazonRiver/IMP_results/dRepALL/RepresentativeBins/ReGes_dictionary.tsv'
# For this workflow, HOST_DB should contain fasta of dereplicated genomes or bins
export HOST_DB='/work/users/smartinezarbas/comparative_analysis/AmazonRiver/IMP_results/dRepALL/RepresentativeBins/dereplicated_genomes/ALL_representative_genomes.fa'
export CRISPR_ELEMENTS_DIR='/scratch/users/smartinezarbas/AmazonRiverCRISPR_MGE/ALL'

export DREP_DONE='no'
export OUTDIR='/scratch/users/smartinezarbas/AmazonRiverCRISPR_MGE/MGE_CrisprHost_link'

snakemake -pf mge_host_link_workflow.done -s workflows/MgeHostLink
 
