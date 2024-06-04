#!/bin/bash
#PBS -l walltime=8:00:00
#PBS -l select=1:ncpus=8:mem=196gb
#PBS -N DEG_NPD
#PBS -o /rds/general/project/ukdrmultiomicsproject/live/MAP_analysis/TREM2_unenriched_scflow/dge/initial_cohort/logs/
#PBS -e /rds/general/project/ukdrmultiomicsproject/live/MAP_analysis/TREM2_unenriched_scflow/dge/initial_cohort/logs/
#PBS -J 1-20

cd $PBS_O_WORKDIR

resource_dir=/rds/general/project/ukdrmultiomicsproject/live/MAP_pipelines

mount_dir=/rds
input_dir=/rds/general/project/ukdrmultiomicsproject/live/MAP_analysis/TREM2_unenriched_scflow/dge/initial_cohort/objects/split_sce_celltype/celltype_sce
input_sce=$(ls $input_dir | head -n $PBS_ARRAY_INDEX | tail -n 1)
input_dir_container=/data/general/project/ukdrmultiomicsproject/live/MAP_analysis/TREM2_unenriched_scflow/dge/initial_cohort/objects/split_sce_celltype/celltype_sce
output_dir=/data/general/project/ukdrmultiomicsproject/live/MAP_analysis/TREM2_unenriched_scflow/dge/initial_cohort/DGE

container_resource_dir=/data/general/project/ukdrmultiomicsproject/live/MAP_pipelines

START=$(date)
echo job started at $START
echo input celltype sce: $input_sce

singularity exec \
-B /rds/general/user/$USER/ephemeral/tmp/:/tmp,/rds/general/user/$USER/ephemeral/tmp/:/var/tmp,$mount_dir:/data \
$resource_dir/snRNAseq/singularity-cache/nfancy-scflow-0.7.4.img \
Rscript $output_dir/run_deg_discrete.r \
--sce $input_dir_container/$input_sce \
--dependent_var NeuropathologicalDiagnosis \
--ref_class Control \
--confounding_vars Sex,Age,PostMortemInterval,BrainRegion,APOEgroup,CD33Group \
--stratification_var TREM2Variant \
--subset_var NeuropathologicalDiagnosis \
--subset_class Control \
--output_dir $output_dir \
--ensembl_mappings $container_resource_dir/snRNAseq/assets/ensembl_mappings_human.tsv

END=$(date)
echo job ended at $END
