#!/bin/bash
#PBS -N infercnv
#PBS -o /rds/general/user/art4017/home/snRNAseq_analysis/log/infercnv.out
#PBS -e /rds/general/user/art4017/home/snRNAseq_analysis/log/infercnv.err
#PBS -l select=1:ncpus=1:mem=20gb
#PBS -l walltime=48:00:00

# tmux new -s snRNAseq_workflow

cd ~/snRNAseq_analysis/

. ~/.bashrc
conda activate nfcore
module load gcc/8.2.0
NXF_OPTS='-Xms1g -Xmx4g'

# # infercnv by sample
nextflow run ../snRNAseq_workflow/04_infercnv.nf \
  -c config/snRNAseq_workflow/oesophageal_10X.config \
  -c config/imperial.config \
  -profile imperial \
  -with-singularity singularity/infercnv.latest.img \
  -w work/snRNAseq_workflow/ 
