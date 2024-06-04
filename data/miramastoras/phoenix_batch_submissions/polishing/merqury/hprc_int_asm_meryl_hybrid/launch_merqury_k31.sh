#!/bin/bash

# Author      : Julian Lucas, juklucas@ucsc.edu
# Description : Launch toil job submission for HPRC's DeepPolisher workflow using Slurm arrays
# Usage       : sbatch launch_hifiasm_array.sh sample_file.csv
#               	sample_file.csv should have a header (otherwised first sample will be skipped)
#					and the sample names should be in the first column

#SBATCH --job-name=merqury_hybrid_hprc_int_asm_k31
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=high_priority
#SBATCH --mail-user=mmastora@ucsc.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --mem=300gb
#SBATCH --threads-per-core=1
#SBATCH --output=merqury_hybrid_k31_submit_logs/merqury_hybrid_k31_submit_%x_%j_%A_%a.log
#SBATCH --time=24:00:00
#SBATCH --array=1-2%1

set -ex

## Pull samples names from CSV passed to script
sample_file=$1

# Read the CSV file and extract the sample ID for the current job array task
# Skip first row to avoid the header
sample_id=$(awk -F ',' -v task_id=${SLURM_ARRAY_TASK_ID} 'NR>1 && NR==task_id+1 {print $1}' "${sample_file}")

# Ensure a sample ID is obtained
if [ -z "${sample_id}" ]; then
    echo "Error: Failed to retrieve a valid sample ID. Exiting."
    exit 1
fi

echo "${sample_id}"

BASE="/private/groups/patenlab/mira/hprc_polishing/hprc_int_asm/meryl_hybrid_k31"
RAW_WG_OUTDIR="/private/groups/patenlab/mira/hprc_polishing/hprc_int_asm/merqury_hybrid_k31/raw_wg/${sample_id}"
POL_WG_OUTDIR="/private/groups/patenlab/mira/hprc_polishing/hprc_int_asm/merqury_hybrid_k31/pol_wg/${sample_id}"
RAW_CONF_OUTDIR="/private/groups/patenlab/mira/hprc_polishing/hprc_int_asm/merqury_hybrid_k31/raw_conf/${sample_id}"
POL_CONF_OUTDIR="/private/groups/patenlab/mira/hprc_polishing/hprc_int_asm/merqury_hybrid_k31/pol_conf/${sample_id}"

mkdir -p $RAW_WG_OUTDIR
mkdir -p $POL_WG_OUTDIR
mkdir -p $RAW_CONF_OUTDIR
mkdir -p $POL_CONF_OUTDIR

# untar hybrid meryl
tar -zxvf ${BASE}/${sample_id}/meryl_hybrid_outputs/${sample_id}.hybrid.meryl.tar.gz -C ${BASE}/${sample_id}/meryl_hybrid_outputs/

# run hybrid merqury for raw assembly, whole genome
time docker run --rm -u `id -u`:`id -g` \
-v /private/groups:/private/groups \
-v ${RAW_WG_OUTDIR}:/data \
juklucas/hpp_merqury:latest merqury.sh \
${BASE}/${sample_id}/meryl_hybrid_outputs/${sample_id}.hybrid.meryl \
/private/groups/hprc/assembly/batch2/${sample_id}/analysis/assembly/${sample_id}.pat.fa.gz \
/private/groups/hprc/assembly/batch2/${sample_id}/analysis/assembly/${sample_id}.mat.fa.gz \
${sample_id}_raw_merqury_hybrid_k31_wg

# run hybrid merqury for polished assembly, whole genome
time docker run --rm -u `id -u`:`id -g` \
-v /private/groups:/private/groups \
-v ${POL_WG_OUTDIR}:/data \
juklucas/hpp_merqury:latest merqury.sh \
${BASE}/${sample_id}/meryl_hybrid_outputs/${sample_id}.hybrid.meryl \
/private/groups/hprc/polishing/batch3/apply_GQ_filter/${sample_id}/applyPolish_pat_outputs/${sample_id}_hap1.polished.fasta \
/private/groups/hprc/polishing/batch3/apply_GQ_filter/${sample_id}/applyPolish_mat_outputs/${sample_id}_hap2.polished.fasta \
${sample_id}_polished_merqury_hybrid_k31_wg

# run hybrid merqury for raw assembly, inside conf regions
time docker run --rm -u `id -u`:`id -g` \
-v /private/groups:/private/groups \
-v ${RAW_CONF_OUTDIR}:/data \
juklucas/hpp_merqury:latest merqury.sh \
${BASE}/${sample_id}/meryl_hybrid_outputs/${sample_id}.hybrid.meryl \
/private/groups/hprc/polishing/batch3/apply_GQ_filter/hprc_polishing_QC_k31/${sample_id}/hprc_polishing_QC_outputs/Raw.hap1.insideConf.subBed.fasta \
/private/groups/hprc/polishing/batch3/apply_GQ_filter/hprc_polishing_QC_k31/${sample_id}/hprc_polishing_QC_outputs/Raw.hap2.insideConf.subBed.fasta \
${sample_id}_raw_merqury_hybrid_k31_conf

# run hybrid merqury for polished assembly, inside conf regions

time docker run --rm -u `id -u`:`id -g` \
-v /private/groups:/private/groups \
-v ${POL_WG_OUTDIR}:/data \
juklucas/hpp_merqury:latest merqury.sh \
${BASE}/${sample_id}/meryl_hybrid_outputs/${sample_id}.hybrid.meryl \
/private/groups/hprc/polishing/batch3/apply_GQ_filter/hprc_polishing_QC_k31/${sample_id}/hprc_polishing_QC_outputs/Polished.hap1.insideConf.subBed.fasta \
/private/groups/hprc/polishing/batch3/apply_GQ_filter/hprc_polishing_QC_k31/${sample_id}/hprc_polishing_QC_outputs/Polished.hap2.insideConf.subBed.fasta \
${sample_id}_polished_merqury_hybrid_k31_conf

EXITCODE=$?
set -e
if [[ "${EXITCODE}" == "0" ]] ; then
    echo "Succeeded."
    rm -rf ${BASE}/${sample_id}/meryl_hybrid_outputs/${sample_id}.hybrid.meryl
else
    echo "Failed."
    exit "${EXITCODE}"
fi
