#!/bin/bash

#PBS -W group_list=fornaxea08
#PBS -q workq
#PBS -l select=1:ncpus=1:ngpus=1:mem=4000mb
#PBS -l place=excl
#PBS -N epistasis
#PBS -J 2501-3500
#PBS -l walltime=12:00:00

cd /home/ghemani/eQTL_2D/v4

module add R
module add cuda
module add cuda-sdk

## PBS_ARRAY_INDEX=${1}
id=${PBS_ARRAY_INDEX}

if [ ! -d "scratch/" ]; then
  mkdir scratch/
fi

if [ ! -d "results/" ]; then
  mkdir results/
fi


bplink="../data/clean_geno_final"
bimfile="${bplink}.bim"
bedfile="${bplink}.bed"
famfile="${bplink}.fam"

egufile="${bplink}.egu"

phenfile="../data/probe_pheno_plink_final.txt"
covfile="../data/covariates_plink_final.txt"
objfile="../data/eqtl2d_objects.RData"

resphen="scratch/resphen${id}"

outfile="results/result${id}.txt"

epigpu="../exe/epiGPU_v9"
R="R"

# Correct phenotype for polygenic effects

${R} --no-save --args ${objfile} ${id} ${resphen} < polygenic.R

# Run the analysis

${epigpu} -A ${egufile} ${outfile} -i 2048 -t i -F 8.3 -I 12.3 -P ${resphen}

gzip ${outfile}



