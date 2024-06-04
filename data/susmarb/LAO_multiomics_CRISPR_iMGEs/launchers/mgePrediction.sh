#!/bin/bash -l
#OAR -n mgePrediction_test
#OAR -l nodes=1/core=6,walltime=48

source /home/users/smartinezarbas/git/gitlab/CRISPR_MGE_pipeline/src/preload_modules.sh

export THREADS='6'

export TS_DIR='/work/users/smartinezarbas/comparative_analysis/AmazonRiver/IMP_results'
export DB_FA_DIR='/work/users/smartinezarbas/comparative_analysis/AmazonRiver/IMP_results/Assemblies'
export TS_SAMPLES='TS2 TD2'
export CRISPR_OUTDIR='/scratch/users/smartinezarbas/AmazonRiverCRISPR_MGE'

snakemake -j 6 -pf plasmid_phage_prediction_workflow.done -s workflows/MgePrediction
#snakemake -pf VirFinder/TS2/TS2_mt_unmapped.done -s workflows/MgePrediction
#snakemake -pf PlasFlow/TS2/TS2_mgmt.done -s workflows/MgePrediction
#snakemake -pf cBar/TS2/TS2_mgmt.done -s workflows/MgePrediction

