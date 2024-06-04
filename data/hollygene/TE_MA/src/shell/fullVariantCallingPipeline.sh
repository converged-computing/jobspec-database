#PBS -S /bin/bash
#PBS -q highmem_q
#PBS -N D1_samples
#PBS -l nodes=1:ppn=1:HIGHMEM
#PBS -l walltime=120:00:00
#PBS -l mem=200gb
#PBS -M hcm14449@uga.edu
#PBS -m abe

#S paradoxus TE MA Quality Control and Mutation Calling pipeline


#location of trimgalore moedule
# trimgalore_module="Trim_Galore/0.4.5-foss-2016b"
#location of fastqc module
fastqc_module="FastQC/0.11.8-Java-1.8.0_144"
#location of BWA module
bwa_module="BWA/0.7.15-foss-2016b"
#location of samtools module
samtools_module="SAMtools/1.6-foss-2016b"
#location of bedtools module
bedtools_module="BEDTools/2.28.0-foss-2018a"
#location of python module
python_module="Python/3.5.2-foss-2016b"
#location of picard module
picard_module="picard/2.4.1-Java-1.8.0_144"
#location of GATK module
GATK_module="GATK/4.0.3.0-Java-1.8.0_144"
deeptools_module="deepTools/3.2.1-foss-2018a-Python-3.6.4"
#location of bamtoBigWig script and accessories
script_location="/scratch/hcm14449/TE_MA_Paradoxus/jbscripts"
#location of bam to bigwig script
bamToBigWig="/scratch/hcm14449/TE_MA_Paradoxus/jbscripts/file_to_bigwig_pe.py"
#location of data to be analyzed
data_dir="/scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/IL_Data/GW_run3/00_fastq"
#location of reference genome to be used
ref_genome="/scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/genome.337.fasta"
ref_genome_base="/scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/genome.337"
#directory reference genome is located in
ref_genome_dir="/scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/"
#where should the output be sent
output_directory="/scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Out/RedoApril2020/D1"
mkdir /scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Out/RedoApril2020
mkdir $output_directory
#location of data to be used in the analysis
raw_data="/scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/AllFastas"
unmapped_bams="/scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/AllFastas/unmapped_bams"
mapped_bams="/scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/AllFastas/mapped_bams"

# cd ${output_directory}
# rm *
# mkdir ${output_directory}

module load ${picard_module}
module load ${bwa_module}
module load ${samtools_module}
module load ${GATK_module}

#######################################################################################
# create a uBAM file
#######################################################################################

for file in ${raw_data}/*_R1_001.fastq.gz

do

FBASE=$(basename $file _R1_001.fastq.gz)
BASE=${FBASE%_R1_001.fastq.gz}
java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144" -jar  \
/usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144/picard.jar FastqToSam \
    FASTQ=${raw_data}/${BASE}_R1_001.fastq.gz \
    FASTQ2=${raw_data}/${BASE}_R2_001.fastq.gz  \
    OUTPUT=${unmapped_bams}/${BASE}_fastqtosam.bam \
    READ_GROUP_NAME=${BASE} \
    SAMPLE_NAME=${BASE} \
    PLATFORM=illumina \
    SEQUENCING_CENTER=GGBC

done




for file in ${raw_data}/*R1.fq.gz

do
  FBASE=$(basename $file R1.fq.gz)
  BASE=${FBASE%R1.fq.gz}
java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144" -jar  \
/usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144/picard.jar FastqToSam \
    FASTQ=${raw_data}/${BASE}R1.fq.gz \
    FASTQ2=${raw_data}/${BASE}R2.fq.gz \
    OUTPUT=${unmapped_bams}/${BASE}_fastqtosam.bam \
    READ_GROUP_NAME=${BASE} \
    SAMPLE_NAME=${BASE} \
    PLATFORM=illumina \
    SEQUENCING_CENTER=GGBC

done

#######################################################################################
# mark Illumina adapters
#######################################################################################

mkdir ${unmapped_bams}/TMP

for file in ${unmapped_bams}/*_fastqtosam.bam

do

FBASE=$(basename $file _fastqtosam.bam)
BASE=${FBASE%_fastqtosam.bam}

java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144" -jar  \
/usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144/picard.jar MarkIlluminaAdapters \
I=${unmapped_bams}/${BASE}_fastqtosam.bam \
O=${unmapped_bams}/${BASE}_markilluminaadapters.bam \
M=${unmapped_bams}/${BASE}_markilluminaadapters_metrics.txt \
TMP_DIR=${unmapped_bams}/TMP \
USE_JDK_DEFLATER=true \
USE_JDK_INFLATER=true

done


#######################################################################################
# #
# #
module load ${picard_module}
#
#
for file in ${unmapped_bams}/*_markilluminaadapters.bam

do

FBASE=$(basename $file _markilluminaadapters.bam)
BASE=${FBASE%_markilluminaadapters.bam}

time java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144" -jar  \
/usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144/picard.jar ValidateSamFile \
      I=${unmapped_bams}/${BASE}_markilluminaadapters.bam \
      MODE=VERBOSE

done

#######################################################################################
# convert BAM to FASTQ and discount adapter sequences using SamToFastq
#######################################################################################

for file in ${output_directory}/*_markilluminaadapters.bam

do

FBASE=$(basename $file _markilluminaadapters.bam)
BASE=${FBASE%_markilluminaadapters.bam}

java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144" -jar  \
/usr/local/apps/eb/picard/picard/2.16.0-Java-1.8.0_144 SamToFastq \
I=${raw_data}/${BASE}_markilluminaadapters.bam \
FASTQ=${raw_data}/${BASE}_samtofastq_interleaved.fq \
CLIPPING_ATTRIBUTE=XT \
CLIPPING_ACTION=2 \
INTERLEAVE=true \
NON_PF=true \
TMP_DIR=${raw_data}/TMP

done


#######################################################################################
# Piped Command: works: aligns samples to reference genome. Output is a .sam file
#######################################################################################

 #index the ref genome
bwa index ${ref_genome}
#
for file in ${raw_data}/*_samtofastq_interleaved.fq

do

FBASE=$(basename $file _samtofastq_interleaved.fq)
BASE=${FBASE%_samtofastq_interleaved.fq}

bwa mem -M -p -t 12 ${ref_genome} ${raw_data}/${BASE}_samtofastq_interleaved.fq > ${output_directory}/${BASE}_bwa_mem.sam



done

java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144" -jar  \
/usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144/picard.jar CreateSequenceDictionary \
      R=${ref_genome} \
      O=${ref_genome_dir}/genome.337.dict


# Piped command: SamToFastq, then bwa mem, then MergeBamAlignment
mkdir ${mapped_bams}

for file in ${unmapped_bams}/*_markilluminaadapters.bam

do

FBASE=$(basename $file _markilluminaadapters.bam)
BASE=${FBASE%_markilluminaadapters.bam}
java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144" -jar  \
/usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144/picard.jar SamToFastq \
I=${unmapped_bams}/${BASE}_markilluminaadapters.bam \
FASTQ=/dev/stdout \
CLIPPING_ATTRIBUTE=XT CLIPPING_ACTION=2 INTERLEAVE=true NON_PF=true \
TMP_DIR=${unmapped_bams}/TMP | \
bwa mem -M -t 7 -p ${ref_genome} /dev/stdin| \
java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144" -jar  \
/usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144/picard.jar MergeBamAlignment \
ALIGNED_BAM=/dev/stdin \
UNMAPPED_BAM=${unmapped_bams}/${BASE}_fastqtosam.bam \
OUTPUT=${unmapped_bams}/${BASE}_pipedNewRef.bam \
R=${ref_genome} CREATE_INDEX=true ADD_MATE_CIGAR=true \
CLIP_ADAPTERS=false CLIP_OVERLAPPING_READS=true \
INCLUDE_SECONDARY_ALIGNMENTS=true MAX_INSERTIONS_OR_DELETIONS=-1 \
PRIMARY_ALIGNMENT_STRATEGY=MostDistant ATTRIBUTES_TO_RETAIN=XS \
TMP_DIR=${unmapped_bams}/TMP

done


# # ###################################################################################################
# # ## Picard to mark duplicates
# # ###################################################################################################

#
# # ###################################################################################################



for file in ${unmapped_bams}/*.sorted.bam

do

FBASE=$(basename $file .sorted.bam)
BASE=${FBASE%.sorted.bam}

time java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144" -jar  \
/usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144/picard.jar MarkDuplicates \
REMOVE_DUPLICATES=TRUE \
I=${unmapped_bams}/${BASE}.sorted.bam \
O=${mapped_bams}/${BASE}_removedDuplicates.bam \
M=${unmapped_bams}/${BASE}_removedDupsMetrics.txt

done


#
# ###################################################################################################
# # Using GATK HaplotypeCaller in GVCF mode
# # apply appropriate ploidy for each sample
# # will need to do this separtely for haploid and diploid samples
# ###################################################################################################
# ###################################################################################################
# #
module load ${GATK_module}

# D1 samples
mkdir ${mapped_bams}/D1

cp ${mapped_bams}/*D1* ${mapped_bams}/D1



for file in ${mapped_bams}/D1/${BASE}*_removedDuplicates.bam

do

FBASE=$(basename $file _removedDuplicates.bam)
BASE=${FBASE%_removedDuplicates.bam}

time gatk HaplotypeCaller \
     -R ${ref_genome} \
     -ERC GVCF \
     -I ${mapped_bams}/D1/${BASE}_removedDuplicates.bam \
     -ploidy 2 \
     -O ${mapped_bams}/D1/${BASE}_variants.g.vcf

done


# ###################################################################################################
### Combine gVCFs before joint genotyping
# ###################################################################################################


time gatk CombineGVCFs \
 -O ${mapped_bams}/D1/D1_cohortNewRef.g.vcf \
 -R ${ref_genome} \
 --variant ${mapped_bams}/D1/D1-A__variants.g.vcf \
 --variant ${mapped_bams}/D1/HM-D1-10_variants.g.vcf \
 --variant ${mapped_bams}/D1/HM-D1-11_variants.g.vcf \
 --variant ${mapped_bams}/D1/HM-D1-12_variants.g.vcf \
 --variant ${mapped_bams}/D1/HM-D1-13_variants.g.vcf \
 --variant ${mapped_bams}/D1/HM-D1-14_variants.g.vcf \
 --variant ${mapped_bams}/D1/HM-D1-15_variants.g.vcf \
 --variant ${mapped_bams}/D1/HM-D1-16_variants.g.vcf


###################################################################################################
### Jointly genotype 8 random samples to identify consensus sequences
###################################################################################################

time gatk GenotypeGVCFs \
        -R ${ref_genome} \
        --variant ${mapped_bams}/D1/D1_cohortNewRef.g.vcf \
        -O ${mapped_bams}/D1/D1_variants_8SamplesNewRef.vcf


# ###################################################################################################
# ## Recalibrate base quality scores in all samples to mask any likely consensus variants
# ###################################################################################################
#
for file in ${mapped_bams}/D1/${BASE}*_removedDuplicates.bam

do

FBASE=$(basename $file _removedDuplicates.bam)
BASE=${FBASE%_removedDuplicates.bam}

time gatk BaseRecalibrator \
-I ${mapped_bams}/D1/${BASE}_removedDuplicates.bam \
--known-sites ${mapped_bams}/D1/D1_variants_8SamplesNewRef.vcf \
-O ${mapped_bams}/D1/${BASE}_recal_data.table \
-R ${ref_genome}

done


# ###################################################################################################
# ## Apply BQSR to bam files
# ###################################################################################################
#
for file in ${mapped_bams}/D1/${BASE}*_removedDuplicates.bam

      do
        FBASE=$(basename $file _removedDuplicates.bam)
        BASE=${FBASE%_removedDuplicates.bam}


        gatk ApplyBQSR \
           -R ${ref_genome} \
           -I ${mapped_bams}/D1/${BASE}_removedDuplicates.bam \
           -bqsr ${mapped_bams}/D1/${BASE}_recal_data.table \
           -O ${mapped_bams}/D1/${BASE}_recalibratedNewRef.bam

        done


  ###################################################################################################
  ## Run HaplotypeCaller again on recalibrated samples
  ###################################################################################################
  ###################################################################################################
  #
        module load ${GATK_module}

        # D1 samples

for file in ${mapped_bams}/D1/${BASE}*_recalibratedNewRef.bam

do

FBASE=$(basename $file _recalibratedNewRef.bam)
BASE=${FBASE%_recalibratedNewRef.bam}
time gatk HaplotypeCaller \
-R ${ref_genome} \
-ERC GVCF \
-I ${mapped_bams}/D1/${BASE}_recalibratedNewRef.bam \
-ploidy 2 \
-O ${mapped_bams}/D1/${BASE}_variants.Recal.g.vcf

done



          ###################################################################################################
          ## Combine gvcfs
          ###################################################################################################
          ###################################################################################################
          #
          #
  module load ${GATK_module}
#
time gatk CombineGVCFs \
-R ${ref_genome} \
-O ${mapped_bams}/D1/D1_FullCohort.g.vcf \
-V ${mapped_bams}/D1/D1-A-mismatched__variants.g.vcf \
-V ${mapped_bams}/D1/D1-1__variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-2_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-3_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-4_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-5_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/D1-6__variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-7_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-8_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-9_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-10_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-11_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-12_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-13_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-14_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-15_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-16_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-17_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-18_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-19_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-20_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/D1-21__variants.Recal.g.vcf \
-V ${mapped_bams}/D1/D1-22__variants.Recal.g.vcf \
-V ${mapped_bams}/D1/D1-23__variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-24_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-25_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-26_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-27_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-28_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-29_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-30_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-31_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-32_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-33_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/D1-34__variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-35_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-36_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-37_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-38_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-39_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-40_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-42_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/D1-43__variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-44_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-45_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/HM-D1-46_variants.Recal.g.vcf \
-V ${output_directory}/HM-H0-33_variants.Recal.g.vcf \
-V ${mapped_bams}/D1/D1-48__variants.Recal.g.vcf
#
#              ###################################################################################################
#              ## Genotype gVCFs (jointly)
#              ###################################################################################################
#              ###################################################################################################
#
#
time gatk GenotypeGVCFs \
-R ${ref_genome} \
-ploidy 2 \
--variant ${mapped_bams}/D1/D1_FullCohort.g.vcf \
-O ${mapped_bams}/D1/D1_FullCohort.vcf

# ###################################################################################################
# ### Find coverage and put into 10k chunks
# ###################################################################################################
#### bedtools genomecov

module load ${bedtools_module}
# report gives per-base depth across entire genome

bedtools genomecov -d -ibam ${output_directory}/D1-A-mismatched__recalibratedNewRef.bam > ${output_directory}/D1-A_depth.txt

module load ${deeptools_module}


for file in ${raw_data}/${BASE}*_piped.bam

do

FBASE=$(basename $file _piped.bam)
BASE=${FBASE%_piped.bam}
OUT="${BASE}_bamCoverage.sh"
echo "#!/bin/bash" > ${OUT}
echo "#PBS -N ${BASE}_bamCoverage" >> ${OUT}
echo "#PBS -l walltime=12:00:00" >> ${OUT}
echo "#PBS -l nodes=1:ppn=1:AMD" >> ${OUT}
echo "#PBS -q batch" >> ${OUT}
echo "#PBS -l mem=20gb" >> ${OUT}
echo "" >> ${OUT}
echo "cd ${raw_data}" >> ${OUT}
echo "module load ${deeptools_module}" >> ${OUT}
echo "" >> ${OUT}
echo "bamCoverage -b ${raw_data}/${BASE}_piped.bam -o ${output_directory}/${BASE}.bedgraph -of bedgraph -bs 10000" >> ${OUT}
qsub ${OUT}

done


for file in ${raw_data}/${BASE}*_piped.bam

do

FBASE=$(basename $file _piped.bam)
BASE=${FBASE%_piped.bam}
samtools sort ${raw_data}/${BASE}_piped.bam \
-o ${raw_data}/${BASE}.sorted.bam

samtools depth \
${raw_data}/${BASE}.sorted.bam \
|  awk '{sum+=$3} END { print "Average = ",sum/NR}' > ${raw_data}/${BASE}.txt

done
#
# #combine depths with filenames into the same file
find . -type f -name "*.txt" -exec awk '{s=$0};END{if(s)print FILENAME,s}' {} \; > D0_depth.txt


#                  # ################
# # ###################################################################################################
# # ### Filter variants
# # Can easily run these interactively
# # ###################################################################################################
#
########################################################################
#### Remove low and high read depth first
gatk SelectVariants \
-R ${ref_genome} \
-V ${output_directory}/D1_FullCohort.vcf \
-O ${output_directory}/D1_noLow.vcf \
-select 'vc.getGenotype("D20-A_").getDP() > 70'

gatk SelectVariants \
-R ${ref_genome} \
-V ${output_directory}/D1_noLow.vcf \
-O ${output_directory}/D1_noLow_noHigh.vcf \
-select 'vc.getGenotype("D20-A_").getDP() < 188'

low_mappability="/scratch/jc33471/pilon/337/mappability/337_lowmappability.bed"
module load ${bedtools_module}

bedtools sort -i ${low_mappability} > ${output_directory}/337_lowmappability_sorted.bed
bedtools intersect -v -a ${output_directory}/D1_noLow_noHigh.vcf -b ${low_mappability} -header > ${output_directory}/D1_noLow_noHigh_redGem.vcf

awk 'NR==FNR{a[$1,$2]; next} !(($1,$2) in a)' ${output_directory}/D0/D0_noLow_noHigh_redGem.vcf ${output_directory}/D0/D0_noLow_noHigh.vcf > ${output_directory}/D0/GEMremoved.txt


gatk SelectVariants \
-R ${ref_genome} \
-V ${output_directory}/D1_noLow_noHigh_redGem.vcf \
-O ${output_directory}/D1_noLow_noHigh_redGem_AncCalls.vcf \
-select 'vc.getGenotype("D20-A_").isCalled()'


gatk SelectVariants \
-R ${ref_genome} \
-V ${output_directory}/D1_noLow_noHigh_redGem_AncCalls.vcf \
-O ${output_directory}/D1_noLow_noHigh_redGem_AncCalls_NoHets.vcf \
-select '!vc.getGenotype("D20-A_").isHet()'

### Ancestor hets only
gatk SelectVariants \
-R ${ref_genome} \
-V ${output_directory}/D1_noLow_noHigh_redGem_AncCalls.vcf \
-O ${output_directory}/D1_noLow_noHigh_redGem_AncCalls_Hets.vcf \
-select 'vc.getGenotype("D20-A_").isHet()'


### select snps and indels and make into tables
gatk SelectVariants \
   -R ${ref_genome} \
   -V ${output_directory}/D1_noLow_noHigh_redGem_AncCalls_NoHets.vcf \
   -O ${output_directory}/D1_noLow_noHigh_redGem_AncCalls_NoHets_SNPs.vcf \
   --max-nocall-number 0 \
   --exclude-non-variants TRUE \
	 --restrict-alleles-to BIALLELIC \
   -select-type SNP
#
gatk SelectVariants \
   -R ${ref_genome} \
   -V ${output_directory}/D1_noLow_noHigh_redGem_AncCalls_NoHets.vcf \
   -O ${output_directory}/D1_noLow_noHigh_redGem_AncCalls_NoHets_Indels.vcf \
   --max-nocall-number 0 \
	 --exclude-non-variants TRUE \
	 --restrict-alleles-to BIALLELIC \
   -select-type INDEL

gatk SelectVariants \
	 -R ${ref_genome} \
   -V ${output_directory}/D1_noLow_noHigh_redGem_AncCalls_NoHets.vcf \
   -O ${output_directory}/D1_noLow_noHigh_redGem_AncCalls_NoHetsVars.vcf \
   --max-nocall-number 0 \
	 --exclude-non-variants TRUE \
	 --restrict-alleles-to BIALLELIC


gatk VariantsToTable \
	 -V ${output_directory}/D1_noLow_noHigh_redGem_AncCalls_NoHetsVars.vcf \
	 -F CHROM -F POS -F REF -F ALT -F QUAL \
	 -GF AD -GF DP -GF GQ -GF GT \
	 -O ${output_directory}/D1_noLow_noHigh_redGem_AncCalls_NoHets_vars.txt

gatk VariantsToTable \
	-V ${output_directory}/D1_noLow_noHigh_redGem_AncCalls_NoHets_SNPs.vcf \
	-F CHROM -F POS -F REF -F ALT -F QUAL \
	-GF AD -GF DP -GF GQ -GF GT \
	-O ${output_directory}/D1_noLow_noHigh_redGem_AncCalls_NoHets_SNPs.txt

gatk VariantsToTable \
-V ${output_directory}/D1_noLow_noHigh_redGem_AncCalls_NoHets_Indels.vcf \
-F CHROM -F POS -F REF -F ALT -F QUAL \
-GF AD -GF DP -GF GQ -GF GT \
-O ${output_directory}/D1_noLow_noHigh_redGem_AncCalls_NoHets_Indels.txt

#### Remove sites with mappability < 0.9
low_mappability="/scratch/jc33471/pilon/337/mappability/337_lowmappability.bed"
module load ${bedtools_module}

# bedtools sort -i ${low_mappability} > ${output_directory}/337_lowmappability_sorted.bed
bedtools intersect -v -a ${output_directory}/D1_FullCohort.vcf -b ${low_mappability} -header > ${output_directory}/D1_reducedGEM.vcf

# Get only those lines where there is actually a genotype call in the ancestor
gatk SelectVariants \
-R ${ref_genome} \
-V ${output_directory}/D1_FullCohort.vcf \
-O ${output_directory}/D1_FullCohort_AncCalls.vcf \
-select 'vc.getGenotype("D1-A_").isCalled()'

#
# remove all lines in the ancestor that have a heterozygous genotype
gatk SelectVariants \
-R ${ref_genome} \
-V ${output_directory}/D1_FullCohort_AncCalls.vcf \
-O ${output_directory}/D1_FullCohort_AncCalls_NoHets.vcf \
-select '!vc.getGenotype("D1-A_").isHet()'

# filter out sites with low read depth
gatk VariantFiltration \
   -R ${ref_genome} \
   -V ${output_directory}/D1_FullCohort_AncCalls_NoHets.vcf \
   -O ${output_directory}/D1_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBias.vcf \
   --set-filtered-genotype-to-no-call TRUE \
   -G-filter "DP < 10"  -G-filter-name "depthGr10" \
   -filter "MQ < 50.0" -filter-name "MQ50" \
   -filter "SOR < 0.01" -filter-name "strandBias"

#
  # remove filtered sites (these were set to no calls ./.)
   gatk SelectVariants \
   -R ${ref_genome} \
   -V ${output_directory}/D1_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBias.vcf \
   -O ${output_directory}/D1_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil.vcf \
   --exclude-filtered TRUE

   gatk SelectVariants \
   -R ${ref_genome} \
   -V ${output_directory}/D1_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil.vcf \
   -O ${output_directory}/D1_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil_Calls.vcf \
   -select 'vc.getGenotype("D1-A_").isCalled()'

# cd ${output_directory}
#
   gatk SelectVariants \
   -R ${ref_genome} \
   -V ${output_directory}/D1_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil_Calls.vcf \
   -O ${output_directory}/D1_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil_Calls_SNPs.vcf \
   --max-nocall-fraction 0 \
   --exclude-non-variants TRUE \
   -select-type SNP

   gatk SelectVariants \
   -R ${ref_genome} \
   -V ${output_directory}/D1_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil_Calls.vcf \
   -O ${output_directory}/D1_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil_Calls_Indels.vcf \
      --max-nocall-fraction 0.001 \
      -select-type INDEL

# # # #gives a final dataset with only called sites in the Ancestor, no heterozygous sites in the ancestor,
# # # # depth > 10, mapping quality > 50, and strand bias (SOR) > 0.01 (not significant)
# # #
# # # #Variants to table
gatk VariantsToTable \
     -V ${output_directory}/D1_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil_Calls.vcf \
     -F CHROM -F POS -F REF -F ALT -F QUAL \
     -GF AD -GF DP -GF GQ -GF GT \
     -O ${output_directory}/D1_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil_Calls_vars.txt

     gatk VariantsToTable \
          -V ${output_directory}/D1_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil_Calls_SNPs.vcf \
          -F CHROM -F POS -F REF -F ALT -F QUAL \
          -GF AD -GF DP -GF GQ -GF GT \
          -O ${output_directory}/D1_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil_Calls_SNPs.txt

          gatk VariantsToTable \
               -V ${output_directory}/D1_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil_Calls_Indels.vcf \
               -F CHROM -F POS -F REF -F ALT -F QUAL \
               -GF AD -GF DP -GF GQ -GF GT \
               -O ${output_directory}/D1_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil_Calls_Indels.txt

time gatk SelectVariants \
       -R ${ref_genome} \
       -V ${output_directory}/Full_cohort.vcf \
       -O ${output_directory}/test.vcf \
       -sample-expressions '!vc.getGenotype('HM-D1-A').isHet()'

       time gatk SelectVariants \
              -R ${ref_genome} \
              -V ${output_directory}/Full_cohort_VF_SV.vcf \
              -O ${output_directory}/Full_cohort_VF_SV_noNoCalls.vcf \
              --max-nocall-number 45



              gatk VariantsToTable \
                   -V ${output_directory}/Full_cohort_VF_SV_noNoCalls.vcf \
                   -F CHROM -F POS -F REF -F ALT -F QUAL -F DP \
                   -GF AD -GF GT -GF DP \
                   -O ${output_directory}/Full_cohort_VF_SV_noNoCalls.txt

gatk VariantFiltration \
-V ${output_directory}/Full_cohort.vcf \
-O ${output_directory}/Full_cohort_VF.vcf \
--genotype-filter-expression "isHet == 1" \
--genotype-filter-name "isHetFilter"


gatk SelectVariants \
-V ${output_directory}/Full_cohort_VF.vcf \
--set-filtered-gt-to-nocall \
-O ${output_directory}/Full_cohort_VF_SV.vcf
