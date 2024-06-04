#!/bin/bash
#PBS -P xe2
#PBS -q hugemem
#PBS -l ncpus=28
#PBS -l walltime=24:00:00
#PBS -l other=gdata1
#PBS -l mem=800G
#PBS -l wd
#PBS -m abe
#PBS -M kevin.murray@anu.edu.au

. raijin/modules.sh

mkdir -p data/log

snakemake --unlock

snakemake                         \
    -j ${PBS_NCPUS}               \
    --rerun-incomplete            \
    --keep-going                  \
    ${target:-all}                \
    |& tee data/log/snakemake.log \

