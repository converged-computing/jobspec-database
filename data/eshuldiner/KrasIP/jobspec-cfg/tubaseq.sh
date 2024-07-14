#!/usr/bin/env bash

#Wrapper script that runs Tuba-seq analysis pipeline.

# Sample call to this script:
# sh tubaseq.sh <Project_ID> <Parameter_ID> <root directory> <number of samples>

#######################
ml python/3.6.4
module load miniconda/3

PROJECT_ID=$1
PARAMETER_ID=$2
ROOT=$3
NSAMPLES=$4
array_lab="1-$4"

# Sets up directory structure for the project
python3 project_set_up.py "--project=${PROJECT_ID}" "--parameter=${PARAMETER_ID}" "--root=${ROOT}"

# Processes gzipped fastqs in parallel, returns files with raw read counts for tumors.
jid_merge_cluster=$(sbatch --array=$array_lab --parsable run_count_reads_array.sh ${PROJECT_ID} ${PARAMETER_ID} ${ROOT})

# Filtering / clustering step. Takes raw read counts as inputs, removes tumors that don't map to sgIDs from the project, and collapses smaller tumors with highly similar barcodes into larger tumors
jid_merge_cluster2=$(sbatch --dependency=afterany:${jid_merge_cluster} --array=$array_lab --parsable run_filtering.sh ${PROJECT_ID} ${PARAMETER_ID} ${ROOT}) 

# Convert reads to cells based on read count for spike-in cells.
jid_merge_cluster3=$(sbatch --dependency=afterany:${jid_merge_cluster2} --array=$array_lab --parsable run_convert_to_cells.sh ${PROJECT_ID} ${PARAMETER_ID} ${ROOT})

# Process tumors
jid4=$(sbatch --dependency=afterany:${jid_merge_cluster3} --parsable run_tumor_processing.sh ${PROJECT_ID} ${PARAMETER_ID} ${ROOT})
