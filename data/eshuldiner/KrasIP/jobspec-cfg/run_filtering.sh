#!/usr/bin/env bash

# Called by tubaseq.sh, after run_count_reads_array.sh has finished running for all samples.
# Submits job array for the tumor filtering step.


#######################

ml python/3.6.4
module load miniconda/3

arg="${3}tubaseq_inp_files/${1}_${2}.inp"
#Parameters
parameters=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${arg})

python3 filter_tumors.py $parameters

