#!/usr/bin/env bash

#PBS -l select=1:ncpus=2:mem=4gb
#PBS -l walltime=12:00:00
#PBS -j oe

cd $PBS_O_WORKDIR
IGENOMES='/data/bio/igenomes/references'

# Load nextflow module
module add java/11.0.17_8-openjdk nextflow/22.10.7

# Start nextflow
nextflow run nf-core/rnaseq     `# Name of nf-core pipeline` \
    -latest                     `# Always fetch latest so -r doesn't fail` \
    -params-file nf-params.json `# Params file (edited by you)` \
    -resume                     `# Outdated, but allows specifying e.g. hg38 in nf-params.json` \
    -r 3.12.0                   `# Could be outdated, but good practice` \
    --igenomes_base $IGENOMES   `# Outdated genomes, but allows specifying e.g. GRCh37 in nf-params.json, pre-indexed` \
    --publish_dir_mode 'link'   `# Try to link output files from work dirs to save space` \