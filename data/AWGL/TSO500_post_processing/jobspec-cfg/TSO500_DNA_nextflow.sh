#!/bin/bash

#SBATCH --output=TSO500_DNA_nextflow-%j-%N.out
#SBATCH --error=TSO500_DNA_nextflow-%j-%N.err
#SBATCH --partition=high

# Description: #Kick off nextflow script to run TSO500 DNA analysis
# Use: TSO500_DNA_nextflow.sh <PATH TO RAW READS> <PATH AND NAME OF SAMPLES ORDER DNA FILE> <SEQUENCING RUN ID>

#####################################################################
# Set up
#####################################################################

FASTQ_PATH=$1
SAMPLES_ORDER=$2
SEQID=$3

#####################################################################
# Run Command
#####################################################################

. ~/.bashrc
module load anaconda

set +u
conda activate somatic_enrichment_nextflow
set -u

nextflow -C /data/diagnostics/pipelines/somatic_enrichment_nextflow/somatic_enrichment_nextflow-main/config/somatic_enrichment_nextflow.config run /data/diagnostics/pipelines/somatic_enrichment_nextflow/somatic_enrichment_nextflow-main/somatic_enrichment_nextflow.nf \
    --fastqs ${FASTQ_PATH}/\*/\*\{R1.fastq.gz,R2.fastq.gz\} \
    --dna_list ${SAMPLES_ORDER} \
    --publish_dir results \
    --sequencing_run ${SEQID} \
    -with-dag ${SEQID}.png \
    -with-report ${SEQID}.html \
    -work-dir work \
    -resume &> pipeline.log

set +u
conda deactivate
set -u

#Only remove work directory if successful pipeline completion
if [[ `tail -n 1 results/post_processing_finished.txt` == "${SEQID} success!." ]]
then
    rm -r work/
fi
