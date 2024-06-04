#!/bin/bash
#PBS -N diffexp
#PBS -o /rds/general/user/art4017/home/snRNAseq_analysis/log/diffexp.out
#PBS -e /rds/general/user/art4017/home/snRNAseq_analysis/log/diffexp.err
#PBS -l select=1:ncpus=1:mem=20gb
#PBS -l walltime=48:00:00

# tmux new -s snRNAseq_workflow

cd ~/snRNAseq_analysis/

. ~/.bashrc
conda activate nfcore
module load gcc/8.2.0
NXF_OPTS='-Xms1g -Xmx4g'
  
# integrate epithelial cells
nextflow run ../snRNAseq_workflow/07_differential_expression.nf \
  -c config/snRNAseq_workflow/oesophageal_10X.config \
  -c config/imperial.config \
  -profile imperial \
  -with-singularity singularity/snRNAseq_workflow.img \
  -w work/snRNAseq_workflow/ 


