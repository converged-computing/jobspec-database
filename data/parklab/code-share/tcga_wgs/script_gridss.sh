#!/bin/bash
#SBATCH -c 8                               # Request 16 cores
#SBATCH -t 5-00:00                         # Runtime in D-HH:MM format
#SBATCH -p medium                           # Partition to run in
#SBATCH --mem=48G                         # Memory total in MB (for all cores)
#SBATCH -o log/gridss_%j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e log/gridss_%j.err                 # File to which STDERR will be written, including job ID (%j)
#SBATCH --array=2-2                                        # You can change the filenames given with -o and -e to any filenames you'd like


GRIDSS_JAR_PATH=/usr/local/bin/gridss-2.13.2.jar

TUMOR_ID='TCGA-HC-A6HX-01A-11D-A755-36'
REFERENCE_ID='TCGA-HC-A6HX-10A-01D-A755-36'
OUT_PATH=/n/data1/hms/dbmi/park/dglodzik/hmf_gridss_test/${TUMOR_ID}/
GRIDSS_PATH=$OUT_PATH/gridss/

TEMP_FOLDER=/n/scratch/users/d/dg204/temp/

mkdir -p "$OUT_PATH"
mkdir -p "$GRIDSS_PATH"


gridssJvmHeap=44g
otherjvmheap=4g

gridssOutput="$GRIDSS_PATH/$TUMOR_ID.gridss.sv.vcf"
gridssAssembly=$TEMP_FOLDER/$TUMOR_ID.bam
gridssThreads=8
gridssTmpDir=$TEMP_FOLDER

reference=/n/data1/hms/dbmi/park/dglodzik/TCGA/reference/GRCh38.d1.vd1.fa

TUMOR_BAM=/n/data1/hms/dbmi/park/dglodzik/TCGA/5555b571-6e03-4564-a0cf-05996b3f52c5/2f5d2e1e-110d-4058-bfe5-17d40b5f2583_wgs_gdc_realn.bam 
REFERENCE_BAM=/n/data1/hms/dbmi/park/dglodzik/TCGA/a550b72d-4290-4def-be69-ba40f36b71f9/7718ad9b-1ded-4d0c-abe8-3095b73e5cde_wgs_gdc_realn.bam

singularity exec -B /n/scratch,/n/data1 /n/app/singularity/containers/gridss_cgap.sif \
gridss \
-r ${reference} \
--jar  ${GRIDSS_JAR_PATH} \
--jvmheap ${gridssJvmHeap} \
--otherjvmheap ${otherjvmheap} \
--output ${gridssOutput} \
--assembly ${gridssAssembly} \
--threads ${gridssThreads} \
--workingdir ${gridssTmpDir} \
--labels ${TUMOR_ID},${REFERENCE_ID} \
$TUMOR_BAM \
$REFERENCE_BAM
