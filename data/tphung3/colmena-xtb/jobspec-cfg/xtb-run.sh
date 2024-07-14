#!/bin/bash
# Script for running the code with the XTB components, useful for debugging/dev work

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Define the version of models to use
mpnn_dir="$DIR/xtb-atomization-v1/"
search_space="$DIR/G13-filtered.csv"


models=`find $mpnn_dir -name best_model.h5 | sort | head -n 1`
python run.py --mpnn-config-directory $mpnn_dir \
    --mpnn-model-files $models \
    --search-space $search_space \
    --qc-parallelism 1 \
    --sampling-fraction 0.005 \
    --molecules-per-ml-task 20000 \
    --qc-spec xtb $@
