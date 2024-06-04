############Data Backup############
/proj/snic2021-23-324/nobackup/private/BEVPAC_WES
sftp kangwang-delivery05046@grus.uppmax.uu.se
get -r UB-2846

############nextflow###############
#data: cd /castor/project/proj/data
#nextflow: cd /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1

#1) Follow the detailed instructions for Uppmax/Bianca (apart from installing Nextflow and nf-core; see step 2): https://github.com/nf-core/configs/blob/master/docs/uppmax.md
#(if you have installed them, you need to undo any changes, and delete any files - to avoid any conflicts)
#2) Load necessary modules: module load bioinfo-tools Nextflow nf-core
#3) Export the following ENV variables, or add them to your .bashrc file:
#4) Run a test run with one paired sample e.g. from WES, using the target BED file, and e.g. the PON from GATK resources. Here is an example:
#module load bioinfo-tools Nextflow nf-core

#!/bin/bash -l
#SBATCH -A sens2019581
#SBATCH -p node 
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH -J sarak_test
module load bioinfo-tools Nextflow nf-core
cd /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1
export NXF_OFFLINE='TRUE'
export NXF_HOME="/castor/project/proj/nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/"
export PATH=${NXF_HOME}:${PATH}
export NXF_TEMP=$SNIC_TMP
export NXF_LAUNCHER=$SNIC_TMP
export NXF_SINGULARITY_CACHEDIR="/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/"

nextflow run /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/main.nf \
-profile uppmax \
-with-singularity "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/nf-core-sarek-2.7.1.simg" \
--custom_config_base "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/configs" \
--project sens2019581 \
--input "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-data/testdata/test2.tsv" \
--genome GRCh38 \
--germline_resource "/castor/project/proj_nobackup/references/Homo_sapiens/GATK/GRCh38/Annotation/GermlineResource/gnomAD.r2.1.1.GRCh38.PASS.AC.AF.only.vcf.gz" \
--germline_resource_index "/castor/project/proj_nobackup/references/Homo_sapiens/GATK/GRCh38/Annotation/GermlineResource/gnomAD.r2.1.1.GRCh38.PASS.AC.AF.only.vcf.gz.tbi" \
--generate_gvcf \
--pon "/castor/project/proj_nobackup/references/Homo_sapiens/GATK/GRCh38/Annotation/GATKBundle/1000g_pon.hg38.vcf.gz" \
--pon_index "/castor/project/proj_nobackup/references/Homo_sapiens/GATK/GRCh38/Annotation/GATKBundle/1000g_pon.hg38.vcf.gz.tbi" \
--step 'mapping' \
--tools 'mutect2,strelka,manta,haplotypecaller,cnvkit,snpEff' \
--target_bed "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-data/SeqCap_EZ_Exome_v2_hg38_targets.v3.pad.sort.merge.bed" \
--save_bam_mapped \
-resume

################################
###########Test for BEV#########
################################
#upload phenotype#
cd /Users/kangwang/KI/Projects/3.BEVPAC
sftp kangwang-sens2019581@bianca-sftp.uppmax.uu.se:kangwang-sens2019581
##
cp -r /castor/project/proj/data/BEVPAC_WES/UB-2845/210820_A00181_0332_AHGCGFDSX2/SampleSheet.csv /proj/nobackup/sens2019581/wharf/kangwang/kangwang-sens2019581
cp -r /proj/nobackup/sens2019581/wharf/kangwang/kangwang-sens2019581/BEVPAC_samplesheet.tsv /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/BEVPAC-data
cp -r /proj/nobackup/sens2019581/wharf/kangwang/kangwang-sens2019581/Twist_Exome_RefSeq_targets_hg38_100bp_padding.bed /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/BEVPAC-data

#!/bin/bash -l
#SBATCH -A sens2019581
#SBATCH -p core -n 8
#SBATCH -t 24:00:00
#SBATCH -J rm_work
cd /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/
rsync -a --delete work/

###To run nextflow
screen -S sarek
control+A+D
exit 
screen ls
screen -r
screen -X -S 36740.sarek quit
#####################NB#######################
#1.always run with resueme -name, Use the "nextflow log" command to show previous run names.
#2.
#3.check with intermediate files, and edit them if there is something wrong.
#such as 
cp /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results/Preprocessing/TSV/recalibrated.tsv /proj/sens2019581/nobackup/wharf/kangwang/kangwang-sens2019581
sftp kangwang-sens2019581@bianca-sftp.uppmax.uu.se:kangwang-sens2019581
cp /proj/sens2019581/nobackup/wharf/kangwang/kangwang-sens2019581/recalibrated.tsv /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results/Preprocessing/TSV
###############################################
#####################Start#####################
###############################################
module load bioinfo-tools Nextflow/21.04.1 nf-core/1.14 iGenomes/latest
export NXF_OFFLINE='TRUE'
export NXF_HOME="/castor/project/proj/nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/"
export PATH=${NXF_HOME}:${PATH}
export NXF_TEMP=$SNIC_TMP
export NXF_LAUNCHER=$SNIC_TMP
export NXF_SINGULARITY_CACHEDIR="/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/"
# Mapping step
nextflow run /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/main.nf \
--outdir "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results" \
-profile uppmax \
-with-singularity "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/nf-core-sarek-2.7.1.simg" \
--custom_config_base "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/configs" \
-c "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-data/sarek.custom.config" \
--project sens2019581 \
--input "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/BEVPAC-data/BEVPAC_samplesheet.tsv" \
--genome GRCh38 \
--step 'mapping' \
--target_bed "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/BEVPAC-data/Twist_Exome_RefSeq_targets_hg38_100bp_padding.bed" \
--save_bam_mapped \
-resume lethal_engelbart 

# Prepare recalibration
module load bioinfo-tools Nextflow/21.04.1 nf-core/1.14 iGenomes/latest
export NXF_OFFLINE='TRUE'
export NXF_HOME="/castor/project/proj/nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/"
export PATH=${NXF_HOME}:${PATH}
export NXF_TEMP=$SNIC_TMP
export NXF_LAUNCHER=$SNIC_TMP
export NXF_SINGULARITY_CACHEDIR="/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/"
nextflow run /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/main.nf \
--outdir "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results" \
-profile uppmax \
-with-singularity "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/nf-core-sarek-2.7.1.simg" \
--custom_config_base "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/configs" \
-c "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-data/sarek.custom.config" \
--project sens2019581 \
--input "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results/Preprocessing/TSV/duplicates_marked_no_table.tsv" \
--genome GRCh38 \
--step 'prepare_recalibration' \
--target_bed "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/BEVPAC-data/Twist_Exome_RefSeq_targets_hg38_100bp_padding.bed" \
--save_bam_mapped \
-resume jovial_sanger

# Recalibrate
module load bioinfo-tools Nextflow/21.04.1 nf-core/1.14 iGenomes/latest
export NXF_OFFLINE='TRUE'
export NXF_HOME="/castor/project/proj/nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/"
export PATH=${NXF_HOME}:${PATH}
export NXF_TEMP=$SNIC_TMP
export NXF_LAUNCHER=$SNIC_TMP
export NXF_SINGULARITY_CACHEDIR="/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/"
nextflow run /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/main.nf \
--outdir "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results" \
-profile uppmax \
-with-singularity "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/nf-core-sarek-2.7.1.simg" \
--custom_config_base "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/configs" \
-c "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-data/sarek.custom.config" \
--project sens2019581 \
--input "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results/Preprocessing/TSV/duplicates_marked.tsv" \
--genome GRCh38 \
--step 'recalibrate' \
--target_bed "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/BEVPAC-data/Twist_Exome_RefSeq_targets_hg38_100bp_padding.bed" \
--save_bam_mapped \
-resume distraught_poincare

#generate index
cd /castor/project/proj_nobackup/references/Homo_sapiens/GATK/GRCh38/Annotation/GATKBundle
module load tabix
tabix -p vcf 1000g_pon.hg38.vcf.gz

# Variant calling: Haplotypecaller
nextflow run /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/main.nf \
-profile uppmax \
-with-singularity "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/nf-core-sarek-2.7.1.simg" \
--custom_config_base "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/configs" \
-c "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-data/sarek.custom.config" \
--project sens2019581 \
--input "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results/Preprocessing/TSV/recalibrated.tsv" \
--genome GRCh38 \
--generate_gvcf \
--pon "/castor/project/proj_nobackup/references/Homo_sapiens/GATK/GRCh38/Annotation/GATKBundle/1000g_pon.hg38.vcf.gz" \
--pon_index "/castor/project/proj_nobackup/references/Homo_sapiens/GATK/GRCh38/Annotation/GATKBundle/1000g_pon.hg38.vcf.gz.tbi" \
--step 'variant_calling' \
--tools 'haplotypecaller' \
--target_bed "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/BEVPAC-data/Twist_Exome_RefSeq_targets_hg38_100bp_padding.bed" \
-resume adoring_brown

# Variant calling: MuTect2
module load bioinfo-tools Nextflow/21.04.1 nf-core/1.14 iGenomes/latest
export NXF_OFFLINE='TRUE'
export NXF_HOME="/castor/project/proj/nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/"
export PATH=${NXF_HOME}:${PATH}
export NXF_TEMP=$SNIC_TMP
export NXF_LAUNCHER=$SNIC_TMP
export NXF_SINGULARITY_CACHEDIR="/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/"
nextflow run /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/main.nf \
--outdir "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results" \
-profile uppmax \
-with-singularity "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/nf-core-sarek-2.7.1.simg" \
--custom_config_base "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/configs" \
-c "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-data/sarek.custom.config" \
--project sens2019581 \
--input "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results/Preprocessing/TSV/recalibrated.tsv" \
--genome GRCh38 \
--germline_resource "$IGENOMES_DATA/Homo_sapiens/GATK/GRCh38/Annotation/GermlineResource/gnomAD.r2.1.1.GRCh38.PASS.AC.AF.only.vcf.gz" \
--germline_resource_index "$IGENOMES_DATA/Homo_sapiens/GATK/GRCh38/Annotation/GermlineResource/gnomAD.r2.1.1.GRCh38.PASS.AC.AF.only.vcf.gz.tbi" \
--pon "/castor/project/proj_nobackup/references/Homo_sapiens/GATK/GRCh38/Annotation/GATKBundle/1000g_pon.hg38.vcf.gz" \
--pon_index "/castor/project/proj_nobackup/references/Homo_sapiens/GATK/GRCh38/Annotation/GATKBundle/1000g_pon.hg38.vcf.gz.tbi" \
--step 'variant_calling' \
--tools 'mutect2' \
--target_bed "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/BEVPAC-data/Twist_Exome_RefSeq_targets_hg38_100bp_padding.bed" \
-resume pensive_bhabha


# Variant calling: Manta (first run Manta, then Strelka ;according to Best Practices)
module load bioinfo-tools Nextflow/21.04.1 nf-core/1.14 iGenomes/latest
export NXF_OFFLINE='TRUE'
export NXF_HOME="/castor/project/proj/nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/"
export PATH=${NXF_HOME}:${PATH}
export NXF_TEMP=$SNIC_TMP
export NXF_LAUNCHER=$SNIC_TMP
export NXF_SINGULARITY_CACHEDIR="/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/"
nextflow run /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/main.nf \
--outdir "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results" \
-profile uppmax \
-with-singularity "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/nf-core-sarek-2.7.1.simg" \
--custom_config_base "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/configs" \
-c "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-data/sarek.custom.config" \
--project sens2019581 \
--input "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results/Preprocessing/TSV/recalibrated.tsv" \
--genome GRCh38 \
--step 'variant_calling' \
--tools 'manta' \
--target_bed "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/BEVPAC-data/Twist_Exome_RefSeq_targets_hg38_100bp_padding.bed" \
-resume

# Variant calling: Strelka (first run Manta, then Strelka ;according to Best Practices)
module load bioinfo-tools Nextflow/21.04.1 nf-core/1.14 iGenomes/latest
export NXF_OFFLINE='TRUE'
export NXF_HOME="/castor/project/proj/nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/"
export PATH=${NXF_HOME}:${PATH}
export NXF_TEMP=$SNIC_TMP
export NXF_LAUNCHER=$SNIC_TMP
export NXF_SINGULARITY_CACHEDIR="/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/"
nextflow run /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/main.nf \
--outdir "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results" \
-profile uppmax \
-with-singularity "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/nf-core-sarek-2.7.1.simg" \
--custom_config_base "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/configs" \
-c "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-data/sarek.custom.config" \
--project sens2019581 \
--input "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results/Preprocessing/TSV/recalibrated.tsv" \
--genome GRCh38 \
--step 'variant_calling' \
--tools 'strelka' \
--target_bed "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/BEVPAC-data/Twist_Exome_RefSeq_targets_hg38_100bp_padding.bed" \
-resume intergalactic_nightingale

# Annotate: snpeff/VEP
nextflow run /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/main.nf \
-profile uppmax \
-with-singularity "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/nf-core-sarek-2.7.1.simg" \
--custom_config_base "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/configs" \
-c "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-data/sarek.custom.config" \
--project sens2019581 \
--input "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results/VariantCalling/*/{Manta,Mutect2,Strelka,HaplotypeCaller,HaplotypeCallerGVCF}/*.vcf.gz" \
--genome GRCh38 \
--step 'annotate' \
--tools 'snpeff, vep' \
-resume

# Variant calling: CNVkit
nextflow run /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/main.nf \
--outdir "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results" \
-profile uppmax \
-with-singularity "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/nf-core-sarek-2.7.1.simg" \
--custom_config_base "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/configs" \
-c "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-data/sarek.custom.config" \
--project sens2019581 \
--input "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results/Preprocessing/TSV/recalibrated.tsv" \
--genome GRCh38 \
--step 'variant_calling' \
--tools 'cnvkit' \
--target_bed "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/BEVPAC-data/Twist_Exome_RefSeq_targets_hg38_100bp_padding.bed" \
-resume

# Annotate: snpeff/VEP

nextflow run /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/main.nf \
-profile uppmax \
-with-singularity "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/nf-core-sarek-2.7.1.simg" \
--custom_config_base "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/configs" \
-c "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-data/sarek.custom.config" \
--project sens2019581 \
--input "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-PROMIX_WES/results/VariantCalling/*/{Manta,Mutect2,Strelka,VarDict}/*.vcf.gz" \
--genome GRCh38 \
--step 'annotate' \
--tools 'snpeff, vep' \
-resume


###################################
############VCF to MAF#############
###################################
###################################
cd /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/MAF
for file in *.vcf.gz;
do 
ID=${file#*_}
ID=${ID%%vs*}
ID=${ID#*_}
ID=$(basename $ID _).vcf
echo unzip $ID
gunzip -c "$file">"$ID"
done
###################################
###################################
###################################
#!/bin/bash -l
#SBATCH -A snic2021-22-358
#SBATCH -p core
#SBATCH -n 8
#SBATCH -t 24:00:00
#SBATCH -J VCFtoMAF
cd /proj/snic2021-23-324/nobackup/private/tools/mskcc-vcf2maf-754d68a
module load bioinfo-tools
module load miniconda3
module load samtools
module load vep
for file in /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/MAF/*.vcf
do
MAF=$(basename $file vcf)maf
ID=$(basename $file .vcf)
Normal=$(echo $ID | cut -d _ -f1)_Blood
echo Parsing $ID
perl vcf2maf.pl --input-vcf $file \
--ref-fasta /sw/data/igenomes/Homo_sapiens/NCBI/GRCh38/Sequence/WholeGenomeFasta/genome.fa \
--species homo_sapiens --ncbi-build GRCh38 \
--vep-path /sw/bioinfo/vep/99/src/ensembl-vep/ \
--vep-data /sw/data/vep/99 \
--tumor-id $ID \
--normal-id $Normal \
--output-maf /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/MAF/$MAF
done
###################################
###################################
###################################
ls | grep "maf" | wc -l
cd /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/MAF
awk '{print $0"\t"FILENAME}' *.maf >  BEVPAC_Mutect2.txt


#####################

# Variant calling: MuTect2
module load bioinfo-tools Nextflow/21.04.1 nf-core/1.14 iGenomes/latest
export NXF_OFFLINE='TRUE'
export NXF_HOME="/castor/project/proj/nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/"
export PATH=${NXF_HOME}:${PATH}
export NXF_TEMP=$SNIC_TMP
export NXF_LAUNCHER=$SNIC_TMP
export NXF_SINGULARITY_CACHEDIR="/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/"
nextflow run /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/main.nf \
--outdir "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results" \
-profile uppmax \
-with-singularity "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/nf-core-sarek-2.7.1.simg" \
--custom_config_base "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/configs" \
-c "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-data/sarek.custom.config" \
--project sens2019581 \
--input "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results/Preprocessing/TSV/BEV115_recalibrated.tsv" \
--genome GRCh38 \
--germline_resource "$IGENOMES_DATA/Homo_sapiens/GATK/GRCh38/Annotation/GermlineResource/gnomAD.r2.1.1.GRCh38.PASS.AC.AF.only.vcf.gz" \
--germline_resource_index "$IGENOMES_DATA/Homo_sapiens/GATK/GRCh38/Annotation/GermlineResource/gnomAD.r2.1.1.GRCh38.PASS.AC.AF.only.vcf.gz.tbi" \
--pon "/castor/project/proj_nobackup/references/Homo_sapiens/GATK/GRCh38/Annotation/GATKBundle/1000g_pon.hg38.vcf.gz" \
--pon_index "/castor/project/proj_nobackup/references/Homo_sapiens/GATK/GRCh38/Annotation/GATKBundle/1000g_pon.hg38.vcf.gz.tbi" \
--step 'variant_calling' \
--tools 'mutect2' \
--target_bed "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/BEVPAC-data/Twist_Exome_RefSeq_targets_hg38_100bp_padding.bed" \
-resume 

###################################
############VCF to MAF#############
###################################
###################################
cd /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/MAF
cd /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/Una_complement
cp /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/Annotation/*/VEP/Mutect2_filtered_BEV*_VEP.ann.vcf.gz .

for file in *.vcf.gz;
do 
ID=${file#*_}
ID=${ID#*_}
ID=$(basename $ID _).vcf
echo unzip $ID
gunzip -c "$file">"$ID"
done

#!/bin/bash -l
#SBATCH -A snic2021-22-358
#SBATCH -p core
#SBATCH -n 8
#SBATCH -t 24:00:00
#SBATCH -J VCFtoMAF
cd /proj/snic2021-23-324/nobackup/private/tools/mskcc-vcf2maf-754d68a
module load bioinfo-tools
module load miniconda3
module load samtools
module load vep
for file in /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/MAF/*.vcf
do
MAF=$(basename $file vcf)maf
ID=$(basename $file .vcf)
Normal=$(echo $ID | cut -d _ -f1)_Blood
echo Parsing $ID
perl vcf2maf.pl --input-vcf $file \
--ref-fasta /sw/data/igenomes/Homo_sapiens/NCBI/GRCh38/Sequence/WholeGenomeFasta/genome.fa \
--species homo_sapiens --ncbi-build GRCh38 \
--vep-path /sw/bioinfo/vep/99/src/ensembl-vep/ \
--vep-data /sw/data/vep/99 \
--tumor-id $ID \
--normal-id $Normal \
--output-maf /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/MAF/$MAF
done
#####################
######Tumor only#####
#####################
#!/bin/bash -l
#SBATCH -A snic2021-22-358
#SBATCH -p core
#SBATCH -n 8
#SBATCH -t 06:00:00
#SBATCH -J VCFtoMAF
cd /proj/snic2021-23-324/nobackup/private/tools/mskcc-vcf2maf-754d68a
module load bioinfo-tools
module load miniconda3
module load samtools
module load vep
for file in /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/Una_complement/Tumor_only/*.vcf
do
MAF=$(basename $file vcf)maf
ID=$(basename $file .vcf)
echo Parsing $ID
perl vcf2maf.pl --input-vcf $file \
--ref-fasta /sw/data/igenomes/Homo_sapiens/NCBI/GRCh38/Sequence/WholeGenomeFasta/genome.fa \
--species homo_sapiens --ncbi-build GRCh38 \
--vep-path /sw/bioinfo/vep/99/src/ensembl-vep/ \
--vep-data /sw/data/vep/99 \
--tumor-id $ID \
--output-maf /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/Una_complement/Tumor_only/$MAF
done

ls | grep "maf" | wc -l
awk '{print $0"\t"FILENAME}' *.maf >  BEVPAC_Mutect2_TumorOnly.txt
####################
#######For BEV155########
####################
#!/bin/bash -l
#SBATCH -A snic2021-22-358
#SBATCH -p core
#SBATCH -n 8
#SBATCH -t 24:00:00
#SBATCH -J VCFtoMAF
cd /proj/snic2021-23-324/nobackup/private/tools/mskcc-vcf2maf-754d68a
module load bioinfo-tools
module load miniconda3
module load samtools
module load vep
for file in /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/Una_complement/BEV155/*.vcf
do
MAF=$(basename $file vcf)maf
ID=$(basename $file .vcf)
Normal=$(echo $ID | cut -d _ -f1)_Blood
echo Parsing $ID
perl vcf2maf.pl --input-vcf $file \
--ref-fasta /sw/data/igenomes/Homo_sapiens/NCBI/GRCh38/Sequence/WholeGenomeFasta/genome.fa \
--species homo_sapiens --ncbi-build GRCh38 \
--vep-path /sw/bioinfo/vep/99/src/ensembl-vep/ \
--vep-data /sw/data/vep/99 \
--tumor-id $ID \
--normal-id $Normal \
--output-maf /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/Una_complement/BEV155/$MAF
done
############################
##########Strelka###########
############################
cp /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/Annotation/*vs*/VEP/Strelka*vs*snvs_VEP.ann.vcf.gz /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/Una_complement/Strelka
cd /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/Una_complement/Strelka
for file in *.vcf.gz;
do 
ID=${file#*_}
echo $ID
ID=${ID%%vs*}
echo $ID
ID=$(basename $ID _).vcf
echo unzip $ID
gunzip -c "$file">"$ID"
done
###################################
###################################
###################################
#!/bin/bash -l
#SBATCH -A snic2021-22-358
#SBATCH -p core
#SBATCH -n 8
#SBATCH -t 24:00:00
#SBATCH -J VCFtoMAF
cd /proj/snic2021-23-324/nobackup/private/tools/mskcc-vcf2maf-754d68a
module load bioinfo-tools
module load miniconda3
module load samtools
module load vep
for file in cd /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/Una_complement/Strelka/*.vcf
do
MAF=$(basename $file vcf)maf
ID=$(basename $file .vcf)
echo Parsing $ID
perl vcf2maf.pl --input-vcf $file \
--ref-fasta /sw/data/igenomes/Homo_sapiens/NCBI/GRCh38/Sequence/WholeGenomeFasta/genome.fa \
--species homo_sapiens --ncbi-build GRCh38 \
--vep-path /sw/bioinfo/vep/99/src/ensembl-vep/ \
--vep-data /sw/data/vep/99 \
--tumor-id "TUMOR" \
--normal-id "NORMAL" \
--output-maf /proj/snic2021-23-324/nobackup/private/BEVPAC_WES/Una_complement/Strelka/$MAF
done

ls | grep "maf" | wc -l
awk '{print $0"\t"FILENAME}' *.maf >  BEVPAC_Strelka.txt


#################################
#################################
##########ADD Blood##############
#################################
#################################
cd /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/blood_add
module load bioinfo-tools Nextflow/21.04.1 nf-core/1.14 iGenomes/latest
export NXF_OFFLINE='TRUE'
export NXF_HOME="/castor/project/proj/nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/"
export PATH=${NXF_HOME}:${PATH}
export NXF_TEMP=$SNIC_TMP
export NXF_LAUNCHER=$SNIC_TMP
export NXF_SINGULARITY_CACHEDIR="/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/"
nextflow run /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/main.nf \
-profile uppmax \
-with-singularity "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/nf-core-sarek-2.7.1.simg" \
--custom_config_base "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/configs" \
--project sens2019581 \
--input "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/BEVPAC-data/BEVPAC_samplesheet_add.tsv" \
--genome GRCh38 \
--target_bed "/proj/nobackup/sens2019581/wharf/kangwang/kangwang-sens2019581/Twist_Comprehensive_Exome_Covered_Targets_GRCh38.pad.sort.merge.bed" \
--save_bam_mapped \
-resume 

# Variant calling: MuTect2
module load bioinfo-tools Nextflow/21.04.1 nf-core/1.14 iGenomes/latest
export NXF_OFFLINE='TRUE'
export NXF_HOME="/castor/project/proj/nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/"
export PATH=${NXF_HOME}:${PATH}
export NXF_TEMP=$SNIC_TMP
export NXF_LAUNCHER=$SNIC_TMP
export NXF_SINGULARITY_CACHEDIR="/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/"
nextflow run /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/main.nf \
-profile uppmax \
-with-singularity "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/nf-core-sarek-2.7.1.simg" \
--custom_config_base "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/configs" \
--project sens2019581 \
--input "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/recalibrated_add.tsv" \
--genome GRCh38 \
--germline_resource "$IGENOMES_DATA/Homo_sapiens/GATK/GRCh38/Annotation/GermlineResource/gnomAD.r2.1.1.GRCh38.PASS.AC.AF.only.vcf.gz" \
--germline_resource_index "$IGENOMES_DATA/Homo_sapiens/GATK/GRCh38/Annotation/GermlineResource/gnomAD.r2.1.1.GRCh38.PASS.AC.AF.only.vcf.gz.tbi" \
--pon "/castor/project/proj_nobackup/references/Homo_sapiens/GATK/GRCh38/Annotation/GATKBundle/1000g_pon.hg38.vcf.gz" \
--pon_index "/castor/project/proj_nobackup/references/Homo_sapiens/GATK/GRCh38/Annotation/GATKBundle/1000g_pon.hg38.vcf.gz.tbi" \
--step 'variant_calling' \
--tools 'mutect2' \
--target_bed "/proj/nobackup/sens2019581/wharf/kangwang/kangwang-sens2019581/Twist_Comprehensive_Exome_Covered_Targets_GRCh38.pad.sort.merge.bed" \
-resume 




