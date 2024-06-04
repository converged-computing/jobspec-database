#!/bin/bash

#PBS -lselect=1:ncpus=1:mem=64gb
#PBS -lwalltime=48:0:0
#PBS -J 1-50

OUT_FOLDER=/path/to/output/folder
mkdir -p $OUT_FOLDER

# BirdNet analyse uses os.walk and no need to feed the exact folder
singularity exec \
    --bind $OUT_FOLDER:/output \
    ~/BirdNET-Analyzer/birdnet.sif \
    python3 ~/BirdNET-Analyzer/analyze.py \
        --workers 50 \
        --worker_index $(($PBS_ARRAY_INDEX -1)) \
        --array_job True \
        --slist species_list.txt