#!/bin/bash

#SBATCH --mail-user=email_address
#SBATCH --mail-type=ALL

## CPU Usage
#SBATCH --mem=350G
#SBATCH --cpus-per-task=40
#SBATCH --time=48:00:00
#SBATCH --nodes=1

## Output and Stderr
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.error


##########
# Set up #
##########

# Load singularity
module load singularity
BIN_VERSION="1.1.0"

# Load env for bcftools
ANNOTATEVARIANTS_INSTALL=annotate_variants_dir
source $ANNOTATEVARIANTS_INSTALL/opt/miniconda3/etc/profile.d/conda.sh
conda activate $ANNOTATEVARIANTS_INSTALL/opt/AnnotateVariantsEnvironment

# Pull latest version, if you already have it, this will be skipped
export SINGULARITY_CACHEDIR=$PWD
singularity pull docker://google/deepvariant:deeptrio-"${BIN_VERSION}"

# Number of threads
NSLOTS=$SLURM_CPUS_PER_TASK

# Go to the submission directory (where the sbatch was entered)
cd $SLURM_SUBMIT_DIR
WORKING_DIR=working_dir

## Set working space
mkdir -p $WORKING_DIR
cd $WORKING_DIR

#### GRCh38 #### 
echo "GRCh38 genome"
GENOME=genome_build
FASTA_DIR=fasta_dir
FASTA_FILE=fasta_file

SEQ_TYPE=seq_type
BAM_DIR=$WORKING_DIR
FAMILY_ID=family_id
PROBAND_ID=proband_id
MOTHER_ID=mother_id
FATHER_ID=father_id
PED=$FAMILY_ID.ped

PROBAND_BAM=${PROBAND_ID}_${GENOME}.dupremoved.sorted.bam
FATHER_BAM=${FATHER_ID}_${GENOME}.dupremoved.sorted.bam
MOTHER_BAM=${MOTHER_ID}_${GENOME}.dupremoved.sorted.bam

PROBAND_VCF=${PROBAND_ID}.vcf.gz
FATHER_VCF=${FATHER_ID}.vcf.gz
MOTHER_VCF=${MOTHER_ID}.vcf.gz

PROBAND_GVCF=${PROBAND_ID}.gvcf.gz
FATHER_GVCF=${FATHER_ID}.gvcf.gz
MOTHER_GVCF=${MOTHER_ID}.gvcf.gz


# Run singularity
singularity run -B /usr/lib/locale/:/usr/lib/locale/ \
	-B "${BAM_DIR}":"/bamdir" \
	-B "${FASTA_DIR}":"/genomedir" \
	-B "${OUTPUT_DIR}":"/output" \
	docker://google/deepvariant:deeptrio-"${BIN_VERSION}" \
	/opt/deepvariant/bin/deeptrio/run_deeptrio \
  	--model_type=$SEQ_TYPE \
  	--ref="/genomedir/$FASTA_FILE" \
  	--reads_child="/bamdir/$PROBAND_BAM" \
	--reads_parent1="/bamdir/$FATHER_BAM" \
	--reads_parent2="/bamdir/$MOTHER_BAM" \
	--output_vcf_child="/output/$PROBAND_VCF" \
	--output_vcf_parent1="/output/$FATHER_VCF" \
	--output_vcf_parent2="/output/$MOTHER_VCF" \
	--sample_name_child="${PROBAND_ID}_${GENOME}" \
	--sample_name_parent1="${FATHER_ID}_${GENOME}" \
	--sample_name_parent2="${MOTHER_ID}_${GENOME}" \
  	--num_shards=$NSLOTS \
	--intermediate_results_dir="/output/intermediate_results_dir" \
	--output_gvcf_child="/output/$PROBAND_GVCF" \
	--output_gvcf_parent1="/output/$FATHER_GVCF" \
	--output_gvcf_parent2="/output/$MOTHER_GVCF" 


#GLNexus
/mnt/common/Precision/GLNexus/glnexus_cli -c DeepVariant_unfiltered \
        $PROBAND_GVCF \
        $FATHER_GVCF \
        $MOTHER_GVCF \
        --threads $NSLOTS \
        > ${FAMILY_ID}.glnexus.merged.bcf

bcftools view ${FAMILY_ID}.glnexus.merged.bcf | bgzip -c > ${FAMILY_ID}.merged.vcf.gz


