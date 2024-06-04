#!/bin/sh

####PBS JOB OPTIONS

#PBS -N ATAC_seq_parallel
#PBS -t 1-77%10
#PBS -l mem=8G
#PBS -l nodes=1:ppn=4


############################################################

# Tools
bwa=/home/ngstree/software/ngs/bin/bwa
samtools=/home/ngstree/software/ngs/bin/samtools
bamtools=/home/ngstree/software/ngs/bin/bamtools
TrimGalore=/home/ngstree/software/ngs/bin/trim_galore063
preseq=/home/users/s.slovin/software/preseq_v2.0/preseq
bedGraphToBigWig=/home/ngstree/ngs/bin/bedGraphToBigWig
macs2=/home/ngstree/software/python27/bin/macs2
ataqv=/opt/software/ngs/bin/ataqv
mkarv=/home/ngstree/software/ngs/src/ataqv/src/scripts/mkarv
bedtools=/home/ngstree/software/ngs/bin/bedtools
picard=/home/ngstree/software/ngs/lib/picard-tools-2.9.2/picard.jar


# Scripts, conf and filtering files
## Filtering configuration files, scripts and indexes:
bamtools_filter_config=/home/users/s.slovin/pipeline/atac/filtering/secondary/bamtools_filter_pe_without_size_filtering.json
bampe_rm_orphan=/home/users/s.slovin/pipeline/atac/bampe_rm_orphan.py
bed=/home/users/s.slovin/genome/atac_nextflow/hg38cr.fa.include_regions.bed

## Index files for bigwig construction
chr_sizes=/home/users/s.slovin/genome/atac_nextflow/hg38cr.fa_no_random_chr.sizes

## deeptool and R scripts for QC step
plotCorrelation=/home/ngstree/software/python/3.5.3/bin/plotCorrelation
multiBamSummary=/home/ngstree/software/python/3.5.3/lib/python3.5/site-packages/deeptools/multiBamSummary.py 
plot_macs_qc=/home/users/s.slovin/pipeline/atac/plot_macs_qc.r

## Header files for MultiQC
mlib_peak_count_header=/home/users/s.slovin/pipeline/atac/peakcalling/mlib_peak_count_header.txt
mlib_frip_score_header=/home/users/s.slovin/pipeline/atac/peakcalling/mlib_frip_score_header.txt
jaccrad_score=/home/users/s.slovin/pipeline/atac/QC/jaccrad_score.txt

## ATAQV index files
TSS_info=/home/users/s.slovin/genome/atac_nextflow/hg38cr.tss.bed
autosomes=/home/users/s.slovin/genome/atac_nextflow/hg38cr.fa.autosomes_no_random_chr.bed

## Programming software:
python=/opt/software/python27/bin/python2.7
python3=/opt/software/python3/bin/python3
python3_5=/home/ngstree/software/python/3.5.3/bin/python3.5
Rscript=/opt/software/R/devel/latest/bin/Rscript

#### 
TF=$(awk '{print $1}' /home/users/s.slovin/TF_1_3_from_157_end.txt | awk -v line=$PBS_ARRAYID 'NR==line')
#### 

# paths
wd="/home/users/s.slovin/scratch/ATAC_seq/TF300_ATAC" 
outdir="${wd}/${TF}"
alnDir="/home/novaworkspace/slovin/TF_ATAC/plates_1_3/aln"

mkdir -p ${outdir}                      # output directory for a given condition 
mkdir -p ${outdir}/fastq                # Trimming and fastqc directory 
mkdir -p /home/novaworkspace/slovin/TF_ATAC/plates_1_3/aln                  # Mapping and filtering directory 
mkdir -p ${outdir}/stats                # Staristics and Summary files directory 
mkdir -p ${outdir}/QC                   # QC directory
mkdir -p ${outdir}/BigWig               # BiGWig files directory
mkdir -p ${outdir}/peakcalling          # Peak calling files directory
mkdir -p ${outdir}/IDR_res              # IDR peak results per replicate 
rep1="${TF}_rep1"
rep2="${TF}_rep2"



# Run parameters:
cpus=3 #processors 
score=30 # quality score



# qsub variables:
# TF = ESR1                            # The current analyzed condition - qsub variable 
index=/home/ngsworkspace/references    # Path to BWA reference 
index_base=hg38cr.fa                   # BWA reference prefix 



# Parse software version numbers
if [ ! -e ${wd}/software_versions.txt ] 
then
export JAVA_HOME=/opt/software/java/jdk-11.0.8/
export PATH=$JAVA_HOME/bin:$PATH 
V1=`$TrimGalore --version | grep version`
echo "TrimGalore! version is ${V1}" >> ${wd}/software_versions.txt

V2=`/opt/software/ngs/bin/cutadapt --version`
echo "Cutadapt version is ${V2}" >> ${wd}/software_versions.txt

$samtools --version | head -1 >> ${wd}/software_versions.txt
$bamtools --version | awk 'FNR==2{print $0}' >> ${wd}/software_versions.txt
$bedtools --version >> ${wd}/software_versions.txt

java -jar -Xmx2g /home/ngstree/software/ngs/lib/picard-tools-2.9.2/picard.jar MergeSamFiles &>${wd}/temp.txt
V3=`cat ${wd}/temp.txt | grep Version`
echo "picard MergeSamFiles version is ${V3}" >> ${wd}/software_versions.txt
rm ${wd}/temp.txt

$bwa &>${wd}/temp.txt
V4=`cat ${wd}/temp.txt | grep Versio`
echo "bwa ${V4}" >> ${wd}/software_versions.txt
rm ${wd}/temp.txt

${preseq} &>${wd}/temp.txt
V4=`cat ${wd}/temp.txt | awk 'FNR==2{print $0}'`
echo "preseq ${V4}" >> ${wd}/software_versions.txt
rm ${wd}/temp.txt


V5=`${ataqv} --version`
echo "ATAQV version is ${V5}" >> ${wd}/software_versions.txt

${python} $macs2 &>${wd}/temp.txt  
cat ${wd}/temp.txt >> ${wd}/software_versions.txt
rm ${wd}/temp.txt

fi 

# 1 #
##### Trimming #####
# Nextera adaptor trimming + fastqc

for rep in ${rep1} ${rep2}
do
# files prefix
read1=_R1_001.fastq.gz
read2=_R2_001.fastq.gz

fq1=/mnt/novaseq/slovin/200*/Reads/${rep}_*${read1} #read1
fq2=/mnt/novaseq/slovin/200*/Reads/${rep}_*${read2} #read2

export PATH=$PATH:/opt/software/ngs/bin/
${TrimGalore} --fastqc_args "--outdir ${outdir}/fastq" -o ${outdir}/fastq --paired ${fq1} ${fq2} --nextera --length 30 --cores 3 --path_to_cutadapt /opt/software/ngs/bin/cutadapt 
done

## Arguments Trim Glore:
### --length ${min_length}: remove all seq with length shorter then ${min_length}
### --fastqc_args: Passing extra arguments will automatically invoke FastQC, so --fastqc does not have to be specified separately
### --path_to_cutadapt: path way to cutadapt
### -- nextera: remove adapters of nextera library
### -- cores: numver of course to use - it is recommended not to use more than 8



# 2 #
##### mapping #####
# script for BWA MEM alignment and filtering non primary reads

for rep in ${rep1} ${rep2}
do
# files prefix
read1=_R1_001_val_1.fq.gz
read2=_R2_001_val_2.fq.gz

fq1=${outdir}/fastq/${rep}*${read1} #read1
fq2=${outdir}/fastq/${rep}*${read2} #read2

# Alignment and filtering non primary reads
ionice -c 3 ${bwa} mem -M -t ${cpus} ${index}/${index_base} $fq1 $fq2 | ${samtools} view -b -h -F 0x0100 -o "${alnDir}/${rep}_aligned_reads.bam" - 
rm ${fq1} ${fq2}

ionice -c 3 ${samtools} sort "${alnDir}/${rep}_aligned_reads.bam" -O bam -o "${alnDir}/${rep}.bam" -@ ${cpus}
ionice -c 3 rm "${alnDir}/${rep}_aligned_reads.bam"

# index bam file 
${samtools} index "${alnDir}/${rep}.bam"
# count the alignments for each flag type
${samtools} flagstat "${alnDir}/${rep}.bam"> ${outdir}/stats/${rep}.sorted.bam.flagstat
# short alignment summary
${samtools} idxstats "${alnDir}/${rep}.bam" > ${outdir}/stats/${rep}.sorted.bam.idxstats
# comperhansive alignment statistics
${samtools} stats "${alnDir}/${rep}.bam" > ${outdir}/stats/${rep}.sorted.bam.stats
done

## Arguments - BWA MEME:
### -t    Number of threads
### -M	Mark shorter split hits as secondary (for Picard compatibility).



# 3 #
##### Evaluate library complexity #####
# Estimate the complexity of a genomic sequencing library, equivalent to predicting and estimating the number of redundant reads from a given sequencing depth and how many will be expected from additional sequencing using an initial sequencing experiment
for rep in ${rep1} ${rep2}
do
ionice -c 3 ${preseq} lc_extrap -output ${outdir}/QC/${rep}.ccurve.txt -verbose -bam -pe ${alnDir}/${rep}.bam
done

## Arguments - preseq:
###  lc_extrap computes the expected future yield of distinct reads and bounds on the number of total distinct reads in the library and the associated confidence intervals. 



# 4 #
##### remove dups #####
# Mark PCR/optic duplicates and remove them
for rep in ${rep1} ${rep2}
do
export JAVA_HOME=/opt/software/java/jdk-11.0.8/
export PATH=$JAVA_HOME/bin:$PATH 
# Run markduplicates
java -jar -Xmx2g ${picard} MarkDuplicates INPUT=${alnDir}/${rep}.bam OUTPUT=${alnDir}/${rep}.sorted.uniq.bam ASSUME_SORTED=true REMOVE_DUPLICATES=true METRICS_FILE=${outdir}/QC/${rep}.MarkDuplicates.metrics.txt VALIDATION_STRINGENCY=LENIENT

# index and extract statistics:
${samtools} index ${alnDir}/${rep}.sorted.uniq.bam
${samtools} idxstats ${alnDir}/${rep}.sorted.uniq.bam > ${outdir}/stats/${rep}.sorted.uniq.bam.idxstats
${samtools} flagstat ${alnDir}/${rep}.sorted.uniq.bam >${outdir}/stats/${rep}.sorted.uniq.bam.flagstat
${samtools} stats ${alnDir}/${rep}.sorted.uniq.bam > ${outdir}/stats/${rep}.sorted.uniq.bam.stats

ionice -c 3 rm ${alnDir}/${rep}.bam ${alnDir}/${rep}.bam.bai
done

# 5 #
##### second filtering #####
# Filter out: Duplicated reads, unmapped first mate read, unmapped second mate read,   insert size <38 or >2000bp, reads with >4 mismatches, soft clipped reads, reads align to multiple positions, ENCODE black list, chrMT reads, keep segments that are part of a pair.

for rep in ${rep1} ${rep2}
do
ionice -c 3 ${samtools} view -F 4 -F 8 -F 256 -F 1024 -F 2048 -f2 -q 1 -L $bed -b "${alnDir}/${rep}.sorted.uniq.bam" -@ ${cpus} | ${bamtools} filter -script $bamtools_filter_config | ${samtools} view -h | awk 'substr($0,1,1)=="@" || ($9>= 38 && $9<=2000) || ($9<=-38 && $9>=-2000)' | ${samtools} view -b > ${alnDir}/${rep}.secondary_filtering.sorted.bam

${samtools} index ${alnDir}/${rep}.secondary_filtering.sorted.bam
${samtools} flagstat  ${alnDir}/${rep}.secondary_filtering.sorted.bam >  ${outdir}/stats/${rep}.secondary_filtering.sorted.bam.flagstat
${samtools} idxstats  ${alnDir}/${rep}.secondary_filtering.sorted.bam >  ${outdir}/stats/${rep}.secondary_filtering.sorted.bam.idxstats
${samtools} stats  ${alnDir}/${rep}.secondary_filtering.sorted.bam > ${outdir}/stats/${rep}.secondary_filtering.sorted.bam.stats

ionice -c 3 rm "${alnDir}/${rep}.sorted.uniq.bam" "${alnDir}/${rep}.sorted.uniq.bam.bai"
done

## Arguments:
### filter_params:
#### 0x0400 - duplicated reads 
#### 0x004 the query sequence itself is unmapped
#### 0x0008 the mate is unmapped
#### 0x001 template having multiple segments in sequencing

### bamtools_filter_config:
#### "tag" : "NM:<=4" includ reads with less than 4 mismatches
#### "cigar" : "*S*" filter soft clipped reads

### Samtools view options:
#### -q only include reads with mapping quality >= INT. mapQ = 0 in BWA mean mapped to multiple positions. 
#### -F Do not output alignments with any bits set in INT present in the FLAG field.
#### -f Only output alignments with all bits set in INT present in the FLAG field. 
#### -L Only output alignments overlapping the input BED FILE
#### -b output in the bam format 



# 6 #
##### third filtering #####
# Filter orphan reads and mates aligned to different chromosomes. It seems that although NF has “remove orphan reads” python script, a small number of orphan reads remains in the BAM file. Therefore I also add the -f2 argument to include only properly paired reads

for rep in ${rep1} ${rep2}
do
${samtools} sort -n -o ${alnDir}/${rep}.sortedbyname.bam ${alnDir}/${rep}.secondary_filtering.sorted.bam -@ ${cpus}
/opt/software/python27/bin/python2.7 ${bampe_rm_orphan} ${alnDir}/${rep}.sortedbyname.bam ${alnDir}/${rep}.orphanfilt.bam --only_fr_pairs
ionice -c 3 rm ${alnDir}/${rep}.secondary_filtering.sorted.bam ${alnDir}/${rep}.secondary_filtering.sorted.bam.bai 

${samtools} sort ${alnDir}/${rep}.orphanfilt.bam -o ${alnDir}/${rep}.third_filtering.sorted.bam -@ ${cpus} 
ionice -c 3  rm  ${alnDir}/${rep}.orphanfilt.bam  ${alnDir}/${rep}.sortedbyname.bam

${samtools} index ${alnDir}/${rep}.third_filtering.sorted.bam 
${samtools} flagstat ${alnDir}/${rep}.third_filtering.sorted.bam  > ${outdir}/stats/${rep}.third_filtering.sorted.bam.flagstat
${samtools} idxstats ${alnDir}/${rep}.third_filtering.sorted.bam > ${outdir}/stats/${rep}.third_filtering.sorted.bam.idxstats
${samtools} stats ${alnDir}/${rep}.third_filtering.sorted.bam > ${outdir}/stats/${rep}.third_filtering.sorted.bam.stats
done

## Argumemts - samtools:
### -f2 include only properly paired reads



# 7 #
##### Similarity among biological replicates #####
# Check the Spearman correlations between BAM files before merging and plot it as a heatmap with HCL. First we compute the read coverage for consecutive bins of equal size (10 kilobases by default) to assess genome-wide similarity of BAM files. Then we plot it as a heatmap with HCL. 

export PATH=$PATH:/home/ngstree/software/
ionice -c 3 ${python3_5} ${multiBamSummary} bins --bamfiles ${alnDir}/${rep1}.third_filtering.sorted.bam ${alnDir}/${rep2}.third_filtering.sorted.bam -o ${outdir}/QC/${TF}_rep_similarity.npz

ionice -c 3 ${python3_5} ${plotCorrelation} -in ${outdir}/QC/${TF}_rep_similarity.npz --corMethod spearman --labels ${rep1} ${rep2} --skipZeros --whatToPlot heatmap --plotNumbers -o ${outdir}/QC/heatmap_SpearmanCorr_rep_${TF}.pdf  --outFileCorMatrix ${outdir}/QC/SpearmanCorr_mtx_rep_${TF}.tab

ionice -c 3 rm ${outdir}/QC/${TF}_rep_similarity.npz



# 8 #
##### convert BAM to BEDPE and shift the alignments (nick by Tn5) #####
# BAMPE to BEDPE: paired-end BAM file is converted to a BED file, and then the extension and shift parameters can be used. It is, in a sense, the combination of the typical 'BAM' and 'BAMPE'. This way information such as the insertion length and the other mate alignment is not being ignored.
# Shifting alignments: reads should be shifted + 4 bp and − 5 bp for positive and negative strand respectively, to account for the 9-bp duplication created by DNA repair of the nick by Tn5 transposase and achieve base-pair resolution of TF footprint and motif related analyses.
for rep in ${rep1} ${rep2}
do
ionice -c 3 ${samtools} sort -n -o ${alnDir}/${rep}.name.sorted.bam ${alnDir}/${rep}.third_filtering.sorted.bam -@ ${cpus} # check if I can remmove this step


ionice -c 3 ${bedtools} bamtobed -i ${alnDir}//${rep}.name.sorted.bam -bedpe | awk -v OFS="\t" '{if($9=="+"){print $1,$2+4,$6+4}else if($9=="-"){print $1,$2-5,$6-5}}' > ${alnDir}/${rep}_fragments.bed
ionice -c 3 rm ${alnDir}/${rep}.name.sorted.bam
done


# 9 #
##### Peak calling and FRiP calculation #####
# Narrow peak calling for NFRs and QCs (total number of peaks and FRiP)
for rep in ${rep1} ${rep2}
do
## files 
flagstat_rep=${outdir}/stats/${rep}.third_filtering.sorted.bam.flagstat


## peak calling
#ionice -c 3 ${python} ${macs2} callpeak -t ${alnDir}/${rep}_fragments.bed -n ${rep} -f BED --nomodel --shift -100 --extsize 200 --keep-dup all --outdir  ${outdir}/peakcalling
## peak calling
ionice -c 3 ${python} ${macs2} callpeak -t ${alnDir}/${rep}_fragments.bed -n ${rep} -f BEDPE --outdir ${outdir}/peakcalling


## number of peaks
cat ${outdir}/peakcalling/${rep}_peaks.narrowPeak | wc -l | awk -v OFS='\t' -v name="${rep}" '{ print name, $1 }' | cat $mlib_peak_count_header - > ${outdir}/peakcalling/${rep}_peaks.count_mqc.tsv


## FRiP  (reads in peaks) / (total mapped reads)
READS_IN_PEAKS=`${bedtools} intersect -a ${outdir}/stats/${rep}.third_filtering.sorted.bam -b ${outdir}/peakcalling/${rep}_peaks.narrowPeak -bed -c -f 0.20 | awk -F '\t' '{sum += $NF} END {print sum}'`
grep 'mapped (' $flagstat_rep | awk -v a="$READS_IN_PEAKS" -v OFS='\t' -v name="$rep" '{print name, a/$1}' | cat $mlib_frip_score_header - > ${outdir}/peakcalling/${rep}_peaks.FRiP_mqc.tsv
done

## Argumrnts macs2:
### -n perfix for output files
### -f file format
### keep-dup how to handal duplicate. since we removed all duplicates we will use all 



# 10 #
#####  Jaccard coefficient among biological replicates peaks #####
# Evaluate the similarity of the two biological peakset sets based on the intersections between them. To this end we use the Jaccard statistic to assess the ratio of the intersection of two peak sets to the union of the two peak sets. Specifically, it measures the ratio of the number of intersecting base pairs between two sets to the number of base pairs in the union of the two sets.

# The jaccard tool requires that your data is pre-sorted by chromosome and then by start position
for rep in ${rep1} ${rep2}
do
ionice -c 3 sort -k1,1 -k2,2n ${outdir}/peakcalling/${rep}_peaks.narrowPeak > ${outdir}/peakcalling/${rep}_peaks.sorted.narrowPeak
done


#Run Jaccard statistic
ionice -c 3 ${bedtools} jaccard -a ${outdir}/peakcalling/${TF}_rep1_peaks.sorted.narrowPeak -b ${outdir}/peakcalling/${TF}_rep2_peaks.sorted.narrowPeak > ${outdir}/QC/${TF}_rep1_rep2.jaccard 

cat ${outdir}/QC/${TF}_rep1_rep2.jaccard | awk -v OFS='\t' -v name=${TF} 'FNR == 2 {print name, $3}' | cat $jaccrad_score - > ${outdir}/QC/${TF}_pulled_peaks.jaccrad_score_mqc.tsv ## check 

ionice -c 3 rm  ${outdir}/peakcalling/${TF}_rep1_peaks.sorted.narrowPeak  ${outdir}/peakcalling/${TF}_rep2_peaks.sorted.narrowPeak



# 11 #
##### Summary plot for peak calling #####
# Create QC plot for peak calling: peakcount, peak length distribution, FC distribution (relative to background), FDR distribution, P-value distribution. 
for rep in ${rep1} ${rep2}
do
ionice -c 3 ${Rscript} ${plot_macs_qc} -i ${outdir}/peakcalling/${rep}_peaks.narrowPeak -s ${rep}_peaks -o ${outdir}/QC -p macs_peak.${rep}
done 


# 12 #
#####  IDR #####
peak1=${outdir}/peakcalling/${TF}_rep1_peaks.narrowPeak
peak2=${outdir}/peakcalling/${TF}_rep2_peaks.narrowPeak
IDR_THRESH=0.05

${idr} --samples ${peak1} ${peak2} --input-file-type narrowPeak --output-file ${outdir}/IDR_res/${TF}_IDR.bed --rank p.value --plot --use-best-multisummit-IDR


# Get peaks passing IDR threshold of -log10(p-val)>1.30103 (pval=0.05)

IDR_THRESH_TRANSFORMED=$(awk -v p=${IDR_THRESH} 'BEGIN{print -log(p)/log(10)}')

awk 'BEGIN{OFS="\t"} $12>='"${IDR_THRESH_TRANSFORMED}"' {print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12}' ${outdir}/IDR_res/${TF}_IDR.bed | sort | uniq | sort -k1,1 -k2,2n > ${outdir}/IDR_res/${TF}_IDR_filtered.bed


rm  ${outdir}/IDR_res/${TF}_IDR.bed 




# 13 #
##### ATAQV #####
# Perform ATAC specific QC plots for peak calling and alignments (plots length distribution/TSS enrichment etc.)
for rep in ${rep1} ${rep2}
do
ionice -c 3 ${ataqv} --peak-file ${outdir}/peakcalling/${rep}_peaks.narrowPeak --tss-file $TSS_info --metrics-file ${outdir}/QC/${rep}.ataqv.json --name ${rep} --autosomal-reference-file $autosomes MT ${alnDir}/${rep}.third_filtering.sorted.bam > ${outdir}/QC/${rep}.ataqv.out
done

ionice -c 3 ${python3} ${mkarv} ${outdir}/QC/${TF}_qc.html ${outdir}/QC/${rep1}.ataqv.json ${outdir}/QC/${rep2}.ataqv.json

#rm ${outdir}/QC/${rep1}.ataqv.json ${outdir}/QC/${rep2}.ataqv.json ${outdir}/QC/${rep1}.ataqv.out ${outdir}/QC/${rep2}.ataqv.out


# 14 #
##### Create bigwig files #####
# Create normalized bigwig files

## rep1
flagstat=${outdir}/stats/${rep1}.third_filtering.sorted.bam.flagstat

SCALE_FACTOR=`grep 'mapped (' $flagstat | awk '{print 1000000/$1}'`
echo $SCALE_FACTOR > ${outdir}/BigWig/${rep1}.scale_factor.txt

ionice -c 3 ${bedtools} genomecov -ibam ${alnDir}/${rep1}.third_filtering.sorted.bam -bg -scale $SCALE_FACTOR -pc | sort -k1,1 -k2,2n > ${outdir}/BigWig/${rep1}.bedGraph
${bedGraphToBigWig} ${outdir}/BigWig/${rep1}.bedGraph $chr_sizes  ${outdir}/BigWig/${rep1}.bigWig
ionice -c 3 rm ${outdir}/BigWig/${rep1}.bedGraph

## rep2
flagstat=${outdir}/stats/${rep2}.third_filtering.sorted.bam.flagstat

SCALE_FACTOR=`grep 'mapped (' $flagstat | awk '{print 1000000/$1}'`
echo $SCALE_FACTOR > ${outdir}/BigWig/${rep2}.scale_factor.txt

ionice -c 3 ${bedtools} genomecov -ibam ${alnDir}/${rep2}.third_filtering.sorted.bam -bg -scale $SCALE_FACTOR -pc | sort -k1,1 -k2,2n > ${outdir}/BigWig/${rep2}.bedGraph
${bedGraphToBigWig} ${outdir}/BigWig/${rep2}.bedGraph $chr_sizes  ${outdir}/BigWig/${rep2}.bigWig

ionice -c 3 rm ${outdir}/BigWig/${rep2}.bedGraph


## Arguments = bedtools genmecov 
### -pc: Calculates coverage of intervals from left point of a pair reads to the right point.Works for BAM files only
### -bg: Report depth in BedGraph format
### scale: Scale the coverage by a constant factor. Each coverage value is multiplied by this factor before being reported.

########## End of Pipeline ##########
#####################################