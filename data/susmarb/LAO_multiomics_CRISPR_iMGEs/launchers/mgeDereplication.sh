#!/bin/bash -l
#OAR -n mgeDereplication_test
#OAR -l nodes=1/core=6,walltime=6

source /home/users/smartinezarbas/git/gitlab/CRISPR_MGE_pipeline/src/preload_modules.sh

export THREADS='6'

export TS_DIR='/work/users/smartinezarbas/comparative_analysis/AmazonRiver/IMP_results'
export DB_FA_DIR='/work/users/smartinezarbas/comparative_analysis/AmazonRiver/IMP_results/Assemblies'
export TS_SAMPLES='TS2 TD2'

export VIRSORTER_DIR='/scratch/users/smartinezarbas/AmazonRiverCRISPR_MGE/MGE_prediction/VirSorter'
export VIRFINDER_DIR='/scratch/users/smartinezarbas/AmazonRiverCRISPR_MGE/MGE_prediction/VirFinder'
export PLASF_DIR='/scratch/users/smartinezarbas/AmazonRiverCRISPR_MGE/MGE_prediction/PlasFlow'
export CBAR_DIR='/scratch/users/smartinezarbas/AmazonRiverCRISPR_MGE/MGE_prediction/cBar'

export WORK_DIR='/scratch/users/smartinezarbas/AmazonRiverCRISPR_MGE/MGE_dereplication'

snakemake --allow-ambiguity -pf mge_dereplication_workflow.done -s workflows/MgeDereplication

