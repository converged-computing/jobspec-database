#!/usr/bin/env bash

# Called by tubaseq.sh. Submits job array for the read counting step.

#######################

ml python/3.6.4
module load miniconda/3

arg="${3}tubaseq_inp_files/${1}_${2}.inp"

parameters=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${arg})

python3 count_reads.py $parameters
