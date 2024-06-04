#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l pmem=14gb
#PBS -l walltime=00:10:00
#PBS -A open
#PBS -o logs/get_scaling_factors.log.out
#PBS -e logs/get_scaling_factors.log.err
#PBS -t 1-512

module load gcc
module load samtools
module load anaconda3
source activate mittal

# Cross all BED x all BAM to generate heatmap and composite libraries

WRK=/path/to/2022-Mittal_SAGA
BAMDIR=$WRK/data/BAM
CONTROL=$WRK/data/BAM/masterNoTag_20180928.bam
BLACKLIST=$WRK/data/ChexMix_Peak_Filter_List_190612.bed
BIN=$WRK/../bin
FDIR=$WRK/data/NormalizationFactors

WINDOW=50
NFR=$WRK/data/RefPT-YEP/NFR_50bp.bed
NFR03=$WRK/data/RefPT-YEP/NFR_03_50bp.bed
NFR04=$WRK/data/RefPT-YEP/NFR_04_50bp.bed
NFRBoth=$WRK/data/RefPT-YEP/NFR_03-04_50bp.bed

ORIGINAL_SCRIPTMANAGER=$BIN/ScriptManager-v0.13.jar
SCRIPTMANAGER=$BIN/ScriptManager-v0.13-$PBS_ARRAYID.jar
COMPOSITES=$BIN/sum_Col_CDT.pl
SUM=$BIN/sum_Row_CDT.pl

cd $WRK
[ -d logs ] || mkdir logs
[ -d $FDIR ] || mkdir $FDIR
cp $ORIGINAL_SCRIPTMANAGER $SCRIPTMANAGER

# Get BAM filename and type
BAMFILE=`ls $BAMDIR/*.bam |head -n $PBS_ARRAYID | tail -1`;
BAM=`basename $BAMFILE ".bam"`
TYPE=`echo $BAM |cut -d"_" -f2`

# Index BAM file if needed
[ -f $BAMFILE.bai ] || samtools index $BAMFILE

# Use different normalization method depending on target/assay
if [ $TYPE == "H2B" ] || [ $TYPE == "H2BK123ub" ] || [ $TYPE == "H2AZ" ] || \
    [ $TYPE == "H3" ] || [ $TYPE == "H3K4me3" ] || [ $TYPE == "H3K9ac" ] || [ $TYPE == "H3K9me2" ] || [ $TYPE == "H3K9me3" ] || [ $TYPE == "H3K14ac" ] || \
    [ $TYPE == "H3K27ac" ] || [ $TYPE == "H3K36me3" ] || [ $TYPE == "H3K79me3" ] || \
    [ $TYPE == "H4"] || [ $TYPE == "H4R3me2" ] || [ $TYPE == "H4K8ac" ] || [ $TYPE == "H4K12ac" ] || [ $TYPE == "H4K16ac" ] || [ $TYPE == "H4K20me1" ];
then
	echo "Calculate NFR window normalization factors"

	TEMP=$WRK/data/NormalizationFactors/$BAM\_NFRw
	[ -d $TEMP ] || mkdir $TEMP

	SF_FILE=$TEMP\_ScalingFactor.out
	echo $'Sample file:\t'$BAMFILE > $SF_FILE
	echo $'Scaling type:\tNFR_median' >> $SF_FILE
	echo $'Window size(bp):\t'$WINDOW >> $SF_FILE

	# All genes
	BASE=$TEMP/all
	BEDFILE=$NFR
	java -jar $SCRIPTMANAGER read-analysis tag-pileup $BEDFILE $BAMFILE \
		-s 15 --combined -o $BASE\_composite.out -M $BASE\_CDT
        perl $SUM $BASE\_CDT_combined.cdt $BASE\_tagcount.cdt
	sort -rnk2,2 $BASE\_tagcount.cdt > $BASE\_tagcount_sorted.cdt
	NGENES=`wc -l $BEDFILE | awk '{print $1}'`
	AVERAGE=`cut -f2 $BASE\_tagcount_sorted.cdt | awk -v n=$NGENES -v w=$WINDOW '{Total=Total+$1} END{print n/Total}'`
	echo $'All:\t'$AVERAGE >> $SF_FILE

	BASE=$TEMP/g03
	BEDFILE=$NFR03
	java -jar $SCRIPTMANAGER read-analysis tag-pileup $BEDFILE $BAMFILE \
		-s 15 --combined -o $BASE\_composite.out -M $BASE\_CDT
        perl $SUM $BASE\_CDT_combined.cdt $BASE\_tagcount.cdt
	sort -rnk2,2 $BASE\_tagcount.cdt > $BASE\_tagcount_sorted.cdt
	NGENES=`wc -l $BEDFILE | awk '{print $1}'`
	AVERAGE=`cut -f2 $BASE\_tagcount_sorted.cdt | awk -v n=$NGENES -v w=$WINDOW '{Total=Total+$1} END{print n/Total}'`
	echo $'03_TFO:\t'$AVERAGE >> $SF_FILE

	BASE=$TEMP/g04
	BEDFILE=$NFR04
	java -jar $SCRIPTMANAGER read-analysis tag-pileup $BEDFILE $BAMFILE \
		-s 15 --combined -o $BASE\_composite.out -M $BASE\_CDT
        perl $SUM $BASE\_CDT_combined.cdt $BASE\_tagcount.cdt
	sort -rnk2,2 $BASE\_tagcount.cdt > $BASE\_tagcount_sorted.cdt
	NGENES=`wc -l $BEDFILE | awk '{print $1}'`
	AVERAGE=`cut -f2 $BASE\_tagcount_sorted.cdt | awk -v n=$NGENES -v w=$WINDOW '{Total=Total+$1} END{print n/Total}'`
	echo $'04_UNB:\t'$AVERAGE >> $SF_FILE
	mv $SF_FILE $DIR

	BASE=$TEMP/both
	BEDFILE=$NFRBoth
	java -jar $SCRIPTMANAGER read-analysis tag-pileup $BEDFILE $BAMFILE \
		-s 15 --combined -o $BASE\_composite.out -M $BASE\_CDT
        perl $SUM $BASE\_CDT_combined.cdt $BASE\_tagcount.cdt
	sort -rnk2,2 $BASE\_tagcount.cdt > $BASE\_tagcount_sorted.cdt
	NGENES=`wc -l $BEDFILE | awk '{print $1}'`
	AVERAGE=`cut -f2 $BASE\_tagcount_sorted.cdt | awk -v n=$NGENES -v w=$WINDOW '{Total=Total+$1} END{print n/Total}'`
	echo $'Both:\t'$AVERAGE >> $SF_FILE

	#rm -r $TEMP

elif [[ $TYPE == "polyA-RNA" ]];
then
	echo "Calculate 2 factors for RNAseq"
	#TOTALTAG=`samtools view -H $BAMFILE | grep '@SQ' | cut -d":" -f 3 | awk '{sum+=$1} END{print sum/1000000}'`
	java -jar $SCRIPTMANAGER read-analysis scaling-factor $BAMFILE -f $BLACKLIST -o $WRK/data/NormalizationFactors/$BAM\_Totalb
	java -jar $SCRIPTMANAGER read-analysis scaling-factor $BAMFILE -f $BLACKLIST --ncis -c $CONTROL -w 500 -o $WRK/data/NormalizationFactors/$BAM\_NCISb
else
	#Classic TF procedure
	echo "Calculate classic TF NCIS normalization factors w/ blacklist"
	#java -jar $SCRIPTMANAGER read-analysis scaling-factor $BAMFILE --total-tag -o data/NormalizationFactors/$BAM\_TotalTag
	#java -jar $SCRIPTMANAGER read-analysis scaling-factor $BAMFILE --ncis -c $CONTROL -w 500 -o data/NormalizationFactors/$BAM\_NCIS_w500
	#java -jar $SCRIPTMANAGER read-analysis scaling-factor $BAMFILE --both -c $CONTROL -w 500 -o data/NormalizationFactors/$BAM\_Both_w500
	#java -jar $SCRIPTMANAGER read-analysis scaling-factor $BAMFILE --ncis -c $CONTROL -w 200 -o data/NormalizationFactors/$BAM\_NCIS_w200
	#java -jar $SCRIPTMANAGER read-analysis scaling-factor $BAMFILE --both -c $CONTROL -w 200 -o data/NormalizationFactors/$BAM\_Both_w200
	java -jar $SCRIPTMANAGER read-analysis scaling-factor $BAMFILE -f $BLACKLIST --ncis -c $CONTROL -w 500 -o $WRK/data/NormalizationFactors/$BAM\_NCISb
fi

rm $SCRIPTMANAGER
