#!/bin/bash
#SBATCH -c 8                               # Request 16 cores
#SBATCH -t 1-00:00                        # Runtime in D-HH:MM format
#SBATCH -A park
#SBATCH -p medium                           # Partition to run in
#SBATCH --mem=64G                         # Memory total in MB (for all cores)
#SBATCH -o log/TCGA_%A_%a.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e log/TCGA_%A_%a.err                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --array=2-8                                        # You can change the filenames given with -o and -e to any filenames you'd like


LNO=${SLURM_ARRAY_TASK_ID}

SAMPLE_TABLE_FN=/home/dg204/park_dglodzik/TCGA/ATM_seq_since2022.jobs.tsv
LNO=${SLURM_ARRAY_TASK_ID}
NTH_LINE=$(cat ${SAMPLE_TABLE_FN} | head -${LNO} | tail -1)
# the line should be: tumor ID, tumor bam, normal ID, normal bam, (37|38)
CASE_ID=$(echo $NTH_LINE | awk -F" " '{ print $1}') 
TUMOR_UUID=$(echo $NTH_LINE | awk -F" " '{ print $3}') 
TUMOR_ID=$(echo $NTH_LINE | awk -F" " '{ print $7}') 
TUMOR_FILE_TCGA_ID=$(echo $NTH_LINE | awk -F" " '{ print $9}') 

TUMOR_BAM=$(echo $NTH_LINE | awk -F" " '{ print $5}') 
REFERENCE_UUID=$(echo $NTH_LINE | awk -F" " '{ print $2}') 
REFERENCE_BAM=$(echo $NTH_LINE | awk -F" " '{ print $4}') 
REFERENCE_ID=$(echo $NTH_LINE | awk -F" " '{ print $6}') 
REFERENCE_FILE_TCGA_ID=$(echo $NTH_LINE | awk -F" " '{ print $8}') 

GENOME_VERSION="38gdc"

echo "$TUMOR_ID $TUMOR_BAM $REFERENCE_ID $REFERENCE_BAM $GENOME_VERSION" 

SCRATCH_FOLDER=/n/scratch/users/d/dg204/TCGAbams/

# make a sample folder
CASE_PATH=/n/data1/hms/dbmi/park/dglodzik/TCGA/ATM/${TUMOR_ID}/
mkdir -p $CASE_PATH
OUT_PATH=${CASE_PATH}/tnscope/
mkdir -p $OUT_PATH

REF_DIRECTORY=/n/data1/hms/dbmi/park/dglodzik/home_ref/hmf/

if [ "$GENOME_VERSION" = "38gdc" ]
then
    hmfHetPonLoci=${REF_DIRECTORY}/38/GermlineHetPon.38.vcf.gz
    hmfGcProfile=${REF_DIRECTORY}/38/GC_profile.1000bp.38.cnp
    # this one is different!
    reference=/n/data1/hms/dbmi/park/dglodzik/TCGA/reference/GRCh38.d1.vd1.fa
    hmfBreakendPon=${REF_DIRECTORY}/38/gridss_pon_single_breakend.38.bed
    hmfBreakpointPon=${REF_DIRECTORY}/38/gridss_pon_breakpoint.38.bedpe
    hmfBreakpointHotspot=${REF_DIRECTORY}/38/known_fusions.38.bedpe
    linxGenomeVersion=38
    hmfFragileSites=${REF_DIRECTORY}/38/fragile_sites_hmf.38.csv
    hmfLineElements=${REF_DIRECTORY}/38/line_elements.38.csv
    hmfKnownFusionData=${REF_DIRECTORY}/38/known_fusion_data.38.csv
    path_to_ensembl_data_cache=${REF_DIRECTORY}/38/ensembl/
    driverGenePanel_fn=${REF_DIRECTORY}/38/DriverGenePanel.38.tsv
    somatic_hotspots_fn=${REF_DIRECTORY}/38/KnownHotspots.somatic.38.vcf
fi

cd $SCRATCH_FOLDER


TUMOR_BAM_FP="$SCRATCH_FOLDER/$TUMOR_UUID/$TUMOR_BAM"
NORMAL_BAM_FP="$SCRATCH_FOLDER/$REFERENCE_UUID/$REFERENCE_BAM"

# download tumor
if [ ! -f "$TUMOR_BAM_FP" ]; then
    /n/data1/hms/dbmi/park/dglodzik/TCGA/gdc-client download -m  /n/data1/hms/dbmi/park/dglodzik/TCGA/ATM_seq_since2022.manifest.txt -t /home/dg204/projects/somagenome/data/external/GDCtokens/gdc-user-token.2024-02-23T00_15_03.846Z.txt "$TUMOR_UUID"
fi

# download normal
if [ ! -f "$NORMAL_BAM_FP" ]; then
    /n/data1/hms/dbmi/park/dglodzik/TCGA//gdc-client download -m  /n/data1/hms/dbmi/park/dglodzik/TCGA/ATM_seq_since2022.manifest.txt -t /home/dg204/projects/somagenome/data/external/GDCtokens/gdc-user-token.2024-02-23T00_15_03.846Z.txt "$REFERENCE_UUID"
fi

# run TNScope
export LD_LIBRARY_PATH="/n/data1/hms/dbmi/park/SOFTWARE/Sentieon/sentieon-genomics-202112.06/lib:$LD_LIBRARY_PATH"
export SENTIEON_LICENSE=license.rc.hms.harvard.edu:8990
export SENTIEON_INSTALL_DIR=/n/data1/hms/dbmi/park/SOFTWARE/Sentieon/sentieon-genomics-202112.06

dbsnp=/home/dg204/park_dglodzik/ref/CGAP/TNScope/known-sites-snp/known-sites-snp.vcf.gz
pon=/home/dg204/park_dglodzik/ref/CGAP/TNScope/panel_of_normals/panel_of_normal.vcf.gz

# if the TNScope result does not exist
if [ ! -f "$OUT_PATH/output.vcf" ]; then
    cd $OUT_PATH
    /home/dg204/repos/cgap-pipeline-somatic-sentieon/dockerfiles/somatic_sentieon/somatic_sentieon_tumor_normal.sh \
    "$TUMOR_BAM_FP" \
    "$TUMOR_FILE_TCGA_ID" \
    "$NORMAL_BAM_FP" \
    "$REFERENCE_FILE_TCGA_ID" \
    "$reference" \
    "$dbsnp" \
    "$pon"
fi


# separate the VCF file by mutation type
SNV_VCF_FP="$OUT_PATH/${TUMOR_ID}_snv.vcf.gz"
if [ ! -f "$SNV_VCF_FP" ]; then
    conda deactivate
    export LD_LIBRARY_PATH=/n/app/python/3.8.12/lib
    module load gcc/9.2.0 python/3.10.11
    module load samtools/1.15.1
    CGAP_SCRIPT_FOLDER="/home/dg204/repos/cgap-somatic-utils/scr/"
    source /home/dg204/jupytervenv3.10/bin/activate
    $CGAP_SCRIPT_FOLDER/somatic_sentieon_vcf_splitter.py \
        -i "$OUT_PATH/output.vcf"  \
        -p "$OUT_PATH/$TUMOR_ID" \
        -o snv \
        -f
fi

# Hartwig
cd /home/dg204/projects/somagenome/src

/home/dg204/projects/somagenome/src/script_purple2_args.sh \
"$TUMOR_ID" \
"$TUMOR_BAM_FP" \
"$REFERENCE_ID" \
"$NORMAL_BAM_FP" \
"$GENOME_VERSION" \
"" \
"${CASE_PATH}/hmf/"


