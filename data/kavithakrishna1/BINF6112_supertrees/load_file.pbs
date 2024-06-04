#!/bin/bash
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=3:00:00

#load modules
module load python/3.8.2
module load git/2.22.0
module load abyss/2.2.0
module load blast+/2.10.1
module load sepp/4.3.10
module load augustus/3.3.2
module load hmmer/3.3
module load emboss/6.6.0
module load busco/4.1.0

# path configuration
# busco
PATH=$PATH:/apps/augustus/3.3.2/scripts/
export BUSCO_CONFIG_FILE="./BUSCO.config.ini"
export AUGUSTUS_CONFIG_PATH="/home/z5155527/supertrees_BINF6112_2020/augustus_config"

#run the program 
snakemake -j --core 2