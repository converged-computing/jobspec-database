#!/bin/bash


#PBS -N eqtl2d
#PBS -J 1-7339
#PBS -q gpu_queue
#PBS -l walltime=12:00:00,nodes=1

# read in pedigree and phenotype information

PBS_ARRAY_INDEX=${1}
id=${PBS_ARRAY_INDEX}

if [ ! -d "../scratch/" ]; then
  mkdir ../scratch/
fi

bplink="../data/clean_geno_final"
bimfile="${bplink}.bim"
bedfile="${bplink}.bed"
famfile="${bplink}.fam"
covfile="../data/covariates_plink_final.txt"
objfile="../data/clean_data_objects.RData"
amatrix="../data/clean"
gctaphen="../scratch/gctaphen"

outfile="result${id}.txt"

epigpu="../exe/epiGPU"
gcta="../exe/gcta64_test_new"
R="/clusterdata/apps/R-2.14/bin/R"


# Correct phenotype for polygenic effects
# This will make a file called gcta.phen.${id}

${R} --no-save --args ${objfile} ${id} ${gctaphen}${id} < setup_gcta.R

${gcta} --reml --grm ${amatrix} --pheno ${gctaphen}${id} --reml-pred-rand --qcovar ${covfile} --out ${gctaphen}${id}


# Create the .fam file

${R} --no-save --args ${famfile} ${gctaphen}${id}.indi.blp < setup_epigpu.R

# Run the analysis

${epigpu} -A ${bedfile} ${bimfile} ${gctaphen}${id}.indi.blp.fam ${outfile} -t i -F 8 -I 12


