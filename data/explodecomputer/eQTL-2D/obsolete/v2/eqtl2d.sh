#!/bin/bash

#PBS -W group_list=fornaxea08
#PBS -q workq
#PBS -l select=1:ncpus=1:ngpus=1:mem=4000mb
#PBS -l place=excl
#PBS -N epistasis
#PBS -J 1001-2000
#PBS -l walltime=12:00:00

cd /home/ghemani/eQTL_2D/v2

module add R
module add cuda
module add cuda-sdk

### PBS_ARRAY_INDEX=${1}
id=${PBS_ARRAY_INDEX}

if [ ! -d "../scratch/" ]; then
  mkdir ../scratch/
fi

bplink="../data/clean_geno_final"
bimfile="${bplink}.bim"
bedfile="${bplink}.bed"
famfile="${bplink}.fam"
egufile="${bplink}.egu"
covfile="../data/covariates_plink_final.txt"
objfile="../data/clean_data_objects.RData"
amatrix="../data/clean"
gctaphen="../scratch/gctaphen"

outfile="result${id}.txt"

epigpu="../exe/epiGPUv1.1p"
gcta="../exe/gcta64_test_new"
#R="/clusterdata/apps/R-2.14/bin/R"
R="R"

# Correct phenotype for polygenic effects
# This will make a file called gcta.phen.${id}

${R} --no-save --args ${objfile} ${id} ${gctaphen}${id} < setup_gcta.R

${gcta} --reml --grm ${amatrix} --pheno ${gctaphen}${id} --reml-pred-rand --qcovar ${covfile} --out ${gctaphen}${id}


# Create the phenotype file (normalise)

${R} --no-save --args ${gctaphen}${id}.indi.blp < setup_epigpu.R

# Run the analysis

${epigpu} -A ${egufile} ${outfile} -i 2048 -t i -F 8 -I 12.3 -P ${gctaphen}${id}.indi.blp.phen


