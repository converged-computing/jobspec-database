#!/bin/bash
#PBS -N run_by_patient_wo_organoids_snRNAseq_workflow
#PBS -o /rds/general/user/art4017/home/snRNAseq_analysis/log/integrate_by_patient_wo_organoids.out
#PBS -e /rds/general/user/art4017/home/snRNAseq_analysis/log/integrate_by_patient_wo_organoids.err
#PBS -l select=1:ncpus=1:mem=20gb
#PBS -l walltime=48:00:00

# tmux new -s snRNAseq_workflow

cd ~/snRNAseq_analysis/

. ~/.bashrc
conda activate nfcore
module load gcc/8.2.0
NXF_OPTS='-Xms1g -Xmx4g'

# # integrate by patient wo organoids
# nextflow run ../snRNAseq_workflow/integrate.nf \
#   -c config/snRNAseq_workflow/oesophageal_10X.config \
#   -c config/imperial.config \
#   -profile imperial \
#   -with-singularity singularity/snRNAseq_workflow.img \
#   --input.run_by_patient_wo_organoids true \
#   --input.run_all false \
#   -w work/snRNAseq_workflow/

# clustering by patient wo organoids
nextflow run ../snRNAseq_workflow/seurat_clustering.nf \
  -c config/snRNAseq_workflow/oesophageal_10X.config \
  -c config/imperial.config \
  -profile imperial \
  -with-singularity singularity/snRNAseq_workflow.img \
  -w work/snRNAseq_workflow/ \
  --input.run_by_patient_wo_organoids true \
  --input.run_by_patient false \
  --input.run_by_sample false \
  --input.run_all false \
  --input.integrate true \
  -resume
