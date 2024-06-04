#!/bin/bash
#PBS -N H1
#PBS -l select=1:ncpus=16:mpiprocs=16:interconnect=1g:mem=200gb
#PBS -l walltime=168:00:00
#PBS -M pengy10@nih.gov

source /etc/profile.d/modules.sh

module purge
module load singularity
module load nextflow
module load python
nextflow run nf-core/mnaseseq -r d7da40328188ee342a63d795905b9b90929dd444 --input design.csv --genome GRCh37 -profile singularity  --seq_center -c nextflow.config --min_insert 120 --max_insert 180 --max_time '96.h' --save_align_intermeds --keep_dups --skip_preseq -resume

#--skip_preseq

## --fragment_size Number of base pairs to extend single-end reads when creating bigWig files
## --min_insert Minimum insert size for filtering of mono-nucleosome paired-end reads.
## --max_insert Maximum insert size for filtering of mono-nucleosome paired-end reads.
