#!/usr/bin/env bash

# Called by tubaseq.sh, after run_filtering.sh has finished running for all samples.
# Submits job array for the step that converts read counts associated with each tumor (i.e., sgID-barcode) 
# to approximate cell counts based on the spike-in cell read counts.

#######################
ml python/3.6.4
module load miniconda/3

arg="${3}tubaseq_inp_files/${1}_${2}.inp"

parameters=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${arg})

python3 convert_to_cells.py $parameters
