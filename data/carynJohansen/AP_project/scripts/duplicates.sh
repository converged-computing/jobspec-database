#!/bin/bash
#PBS -l nodes=1:ppn=12
#PBS -l mem=24GB
#PBS -l walltime=5:00:00
#PBS -N mark_duplicates 
#PBS -M ckj239@nyu.edu
#PBS -j oe
#PBS -t 2-16

module purge

source /home/ckj239/AppliedGenomics/Project/pipeline.conf

cd $dataDIR
 
srr="$(head -$PBS_ARRAYID $sampleInfo | tail -1 | awk '{print $1}')"

echo "Sample Information"
echo "========================"
echo "Array ID:" $PBS_ARRAYID
echo "Home directory:" $homeDIR
echo "Data directory:" $dataDIR
echo "Current directory:" $PWD
echo "SRR sample:" $srr
echo "reference genome:" $reference
echo "========================"

module load picard-tools/1.88
module load samtools/intel/1.3

echo "picard-tools version"
echo "not quite sure how to use picard..."
echo "samtool version"
echo "$(samtools --version)"

#get aligned sams from STAR alignment directory
samDIR=$dataDIR/$srr\_alignment
alignFile=$srr\Aligned.out.sam #I THINK, not sure
sortedFile=$srr\_s.bam
dupFile=$srr\_sm.bam

cd $samDIR

echo "Alignment Information"
echo "========================"
echo "Alignment Directory:"  $samDIR
echo "Current Directory:" $PWD
echo "Alignment File:" $alignFile
echo "Sorted Alignment File:" $sortedFile
echo "Duplicates Marked Alignment File:" $dupFile	

#convert sam to bam
java -Xmx2g -jar /share/apps/picard-tools/1.88/SortSam.jar \
INPUT=$alignFile \
OUTPUT=$sortedFile \
SORT_ORDER=coordinate

__ERR__=$?
echo "picard SortSam error:" $__ERR__

#mark duplicates in bam
#do we want to remove duplicates? let's count first and see how prevelent they are... Also check the paper.

java -Xmx2g -jar /share/apps/picard-tools/1.88/MarkDuplicates.jar \
INPUT=$sortedFile \
OUTPUT=$dupFile \
METRICS_FILE=metrics.txt \
MAX_FILE_HANDLES_FOR_READ_ENDS_MAP=1000
__ERR__=$?
echo "picard-tools MarkDuplicate error:" $__ERR__

#build a bam index
java -Xmx2g -jar /share/apps/picard-tools/1.88/BuildBamIndex.jar \
INPUT=$dupFile

__ERR__=$?
echo "picard BuildBamIndex error:" $__ERR__
echo "========================"

