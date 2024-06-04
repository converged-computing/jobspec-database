#!/bin/bash

#PBS -l walltime=6:00:00
#PBS -l select=1:ncpus=10:mem=96gb
#PBS -N selecive_enrichment_1
#PBS -o /rds/general/user/jmm17/home/ukdri_projects_code/unenriched_TREM2/24-05_selective_enrichment_workflow/logs/
#PBS -e /rds/general/user/jmm17/home/ukdri_projects_code/unenriched_TREM2/24-05_selective_enrichment_workflow/logs/
#PBS -J 1-20

cd $PBS_O_WORKDIR

input_dir="/rds/general/project/ukdrmultiomicsproject/live/MAP_analysis/TREM2_unenriched_scflow/sce_objects/initial_cohort/split_sce_celltype/celltype_sce"

cd $PBS_O_WORKDIR

singularity_container_path="/rds/general/user/jmm17/home/ukdri_projects_resources/images/snrnaseq-selective-enrichment_latest.img"
sce_path=$(ls $input_dir | head -n $PBS_ARRAY_INDEX | tail -n 1)
geneset_path="/project/ukdri_projects_data/unenriched_trem2/gazestani_et_al.qs"
outdir="/data/MAP_analysis/TREM2_unenriched_scflow"
geneset_name="gazestani"

# maybe add option for specifying a list, or list of pairs, to set all the models to use (or just, NPD, abeta, ptau - either model or null, then test with if/else)

START=$(date)

echo job started at $START
singularity exec \
-B /rds/general/ephemeral/user/$USER/ephemeral:/tmp,/rds/general/ephemeral/user/$USER/ephemeral:/var/tmp,/rds/general/user/jmm17/home:/project,/rds/general/project/ukdrmultiomicsproject/live:/data \
$singularity_container_path \
Rscript ~/ukdri_projects_code/toolkit/workflow/aucell_dream/run_selective_enrichment.r \
--sce_path="/data/MAP_analysis/TREM2_unenriched_scflow/sce_objects/initial_cohort/split_sce_celltype/celltype_sce/$sce_path" \
--geneset_path=$geneset_path \
--outdir=$outdir \
--geneset_name=$geneset_name

END=$(date)
echo job ended at $END