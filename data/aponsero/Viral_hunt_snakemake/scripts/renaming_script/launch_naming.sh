#!/bin/bash

#PBS -W group_list=bhurwitz
#PBS -q standard
#PBS -l select=1:ncpus=2:mem=10gb
#PBS -l walltime=01:00:00
#PBS -M aponsero@email.arizona.edu
#PBS -m bea

module load python
source activate viral_env

CONTIG_RAW="/xdisk/bhurwitz/mig2020/rsgrps/bhurwitz/alise/my_scripts/Viral_hunt_snakemake/test/file1.fasta"
SAMPLE="mytest"

SCRIPT="/xdisk/bhurwitz/mig2020/rsgrps/bhurwitz/alise/my_scripts/Viral_hunt_snakemake/scripts"
RUN="$SCRIPT/renaming_script/correct_contig_names.py"

python $RUN -f $CONTIG_RAW -n $SAMPLE






