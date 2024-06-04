#!/bin/bash -l
#OAR -n mgeRemapping_test
#OAR -l nodes=1/core=12,walltime=96

source /home/users/smartinezarbas/git/gitlab/CRISPR_MGE_pipeline/src/preload_modules.sh

export THREADS='3'

export TS_DIR='/work/users/smartinezarbas/comparative_analysis/AmazonRiver/IMP_results'
export TS_SAMPLES='TS2 TD2'

export IGE_DIR='/scratch/users/smartinezarbas/AmazonRiverCRISPR_MGE/MGE_dereplication'

export OUTDIR='/scratch/users/smartinezarbas/AmazonRiverCRISPR_MGE/MGE_remapping'

snakemake -j 4 -pfk mge_remapping_workflow.done -s workflows/MgeRemapping
#snakemake -j 4 -pfk Annotations/ALL_MGEs_annotation.gff -s workflows/MgeRemapping 
#snakemake -j 4 -pfk Index/ALL_MGEs-merged.fa -s workflows/MgeRemapping
#snakemake -j 4 -pfk Annotations/ALL_MGEs_annotation.gff -s workflows/MgeRemapping
#snakemake -j 4 -pfk Calculations/ContigLevel/ALL_mg_contig_depth.txt -s workflows/MgeRemapping
 
