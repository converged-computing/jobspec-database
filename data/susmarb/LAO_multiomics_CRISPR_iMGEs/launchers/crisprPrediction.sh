#!/bin/bash -l
#OAR -n crisprPrediction_test
#OAR -l nodes=1/core=8,walltime=48

source /home/users/smartinezarbas/git/gitlab/CRISPR_MGE_pipeline/src/preload_modules.sh

export THREADS='8'

export TS_DIR='/work/users/smartinezarbas/comparative_analysis/AmazonRiver/IMP_results'
export DB_FA_DIR='/work/users/smartinezarbas/comparative_analysis/AmazonRiver/IMP_results/Assemblies'
export TS_SAMPLES='TS2 TD2'
export CRISPR_OUTDIR='/scratch/users/smartinezarbas/AmazonRiverCRISPR_MGE/CRISPR_prediction'

snakemake -j 8 -pf crispr_workflow.done -s workflows/CrisprPrediction
