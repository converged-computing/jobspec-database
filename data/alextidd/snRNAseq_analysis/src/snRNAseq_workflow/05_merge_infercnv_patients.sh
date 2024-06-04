#!/bin/bash
#PBS -N merge_infercnv_patients
#PBS -o /rds/general/user/art4017/home/snRNAseq_analysis/log/merge_infercnv_patients.out
#PBS -e /rds/general/user/art4017/home/snRNAseq_analysis/log/merge_infercnv_patients.err
#PBS -l select=1:ncpus=1:mem=20gb
#PBS -l walltime=48:00:00

# tmux new -s snRNAseq_workflow

cd ~/snRNAseq_analysis/

. ~/.bashrc
conda activate nfcore
module load gcc/8.2.0
NXF_OPTS='-Xms1g -Xmx4g'

# # infercnv by sample
nextflow run ../snRNAseq_workflow/05_merge_infercnv_patients.nf \
  -c config/snRNAseq_workflow/oesophageal_10X.config \
  -c config/imperial.config \
  -profile imperial \
  -with-singularity singularity/snRNAseq_workflow.img \
  -w work/snRNAseq_workflow/ 
