#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l pmem=14gb
#PBS -l walltime=00:15:00
#PBS -A open
#PBS -o logs/bulk_geo_tracks.log.out
#PBS -e logs/bulk_geo_tracks.log.err
#PBS -t 1-278

module load gcc
module load samtools
module load anaconda3
source activate mittal

# Cross all BED x all BAM to generate heatmap and composite libraries

# Fill in placeholder constants with your directories
WRK=/path/to/2022-Mittal_SAGA
WRK=/storage/home/owl5022/scratch/2022_Mittal/polish/2022-Mittal_SAGA
BAMDIR=$WRK/data/BAM
OUTDIR=$WRK/03_Bulk_Processing/Tracks
MD5=$WRK/03_Bulk_Processing/Checksums
#grep '^@SQ' 11998_Nrg2_i5006_BY4741_-_YPD_-_XO_FilteredBAM.header |cut -f2,3 | sed 's/SN://g' | sed 's/LN://g' | sort -k1,1 -k2,2n > data/sacCer3.chrom.sizes
CHRMSZ=$WRK/data/sacCer3.chrom.sizes

# Script shortcuts
ORIGINAL_SCRIPTMANAGER=$WRK/bin/ScriptManager-v0.13.jar
SCRIPTMANAGER=/path/to/ScriptManager-v0.13.jar
SCRIPTMANAGER=$WRK/bin/ScriptManager-v0.13-$PBS_ARRAYID.jar
COMPOSITE=$WRK/bin/sum_Col_CDT.pl
BGTOBW=$WRK/bin/bedGraphToBigWig

# Determine BAM file for the current job array index
BAMFILE=`ls $BAMDIR/*.bam | head -n $PBS_ARRAYID |tail -1`
#BAMFILE=`ls $BAMDIR/*.bam | grep -f 00_Download_and_preprocessing/stable4.txt |head -n $PBS_ARRAYID |tail -1`
BAM=`basename $BAMFILE ".bam"`
TYPE=`echo $BAM |cut -d"_" -f2`
TEMP=$OUTDIR/$BAM
BASE=$BAM\_read1

cd $WRK
cp $ORIGINAL_SCRIPTMANAGER $SCRIPTMANAGER
[ -f $BAMFILE.bai ] || samtools index $BAMFILE
[ -d $OUTDIR ] || mkdir $OUTDIR
[ -d $TEMP ] || mkdir $TEMP

if [ $TYPE == "H2B" ] || [ $TYPE == "H2BK123ub" ] || [ $TYPE == "H2AZ" ] || [ $TYPE == "H3" ] || [ $TYPE == "H3K4me3" ] || [ $TYPE == "H3K9ac" ] || [ $TYPE == "H3K9me2" ] || [ $TYPE == "H3K9me3" ] || [ $TYPE == "H3K14ac" ] || [ $TYPE == "H3K27ac" ] || [ $TYPE == "H3K36me3" ] || [ $TYPE == "H3K79me3" ] || [ $TYPE == "H4"] || [ $TYPE == "H4R3me2" ] || [ $TYPE == "H4K8ac" ] || [ $TYPE == "H4K12ac" ] || [ $TYPE == "H4K16ac" ] || [ $TYPE == "H4K20me1" ];
then
	echo "Run Nucleosome:  $BAMFILE"
        #FACTOR=`grep 'Scaling factor' $WRK/data/NormalizationFactors/$BAM\_NFRw_ScalingFactors.out | awk -F" " '{print $3}'`
        FACTOR=`grep 'Both:' $WRK/data/NormalizationFactors/$BAM\_NFRw_ScalingFactor.out | awk -F" " '{print $2}'`

	BASE=$BAM\_read1

        java -jar $SCRIPTMANAGER bam-format-converter bam-to-scidx $BAMFILE -o $TEMP/$BASE.raw.tab
elif [ $TYPE == "polyA-RNA" ] || [ $TYPE == "poly-A-RNA" ] || [ $TYPE == "poly-A-NCISb" ] || [ $TYPE == "poly-A-Totalb" ];
then
	echo "Run RNA pileup:  $BAMFILE"
	FACTOR=`grep 'Scaling factor' $WRK/data/NormalizationFactors/$BAM\_NCISb_ScalingFactors.out | awk -F" " '{print $3}'`

	SID=`echo $BAM | awk -F"_" '{print $1}'`
	SUFFIX=`echo $BAM | awk -F"_" '{OFS="_"}{print $3,$4,$5,$6,$7,$8}'`
	NCIS_BASE=$SID\_poly-A-NCISb\_$SUFFIX\_read2
	echo $NCIS_BASE

	java -jar $SCRIPTMANAGER bam-format-converter bam-to-scidx $BAMFILE -2 -o $TEMP/$BASE.raw.tab
else
	echo "Run classic TF pileup: $BAMFILE"
	FACTOR=`grep 'Scaling factor' $WRK/data/NormalizationFactors/$BAM\_NCISb_ScalingFactors.out | awk -F" " '{print $3}'`

	BASE=$BAM\_read1

	java -jar $SCRIPTMANAGER bam-format-converter bam-to-scidx $BAMFILE -o $TEMP/$BASE.raw.tab
fi

grep -v -f $CHRMSZ $TEMP/$BASE.raw.tab > $TEMP/$BASE.filtered.tab
java -jar $SCRIPTMANAGER read-analysis scale-matrix $TEMP/$BASE.filtered.tab -s $FACTOR -r 2 -o $TEMP/$BASE.scaled.tab

# Convert to sense/anti Bedgraph tracks
sed '1d;2d' $TEMP/$BASE.scaled.tab | awk '{OFS="\t"}{FS="\t"}{print $1, $2, $2+1, $3}' |sort -k1,1 -k2,2n > $TEMP/$BASE.forward.bedgraph
sed '1d;2d' $TEMP/$BASE.scaled.tab | awk '{OFS="\t"}{FS="\t"}{print $1, $2, $2+1, $4}' |sort -k1,1 -k2,2n > $TEMP/$BASE.reverse.bedgraph

# Compress to BigWig format
$BGTOBW $TEMP/$BASE.forward.bedgraph $CHRMSZ $OUTDIR/$BASE.forward.bw
$BGTOBW $TEMP/$BASE.reverse.bedgraph $CHRMSZ $OUTDIR/$BASE.reverse.bw

# Get Checksums
md5sum $OUTDIR/$BASE.forward.bw > $MD5/$BASE.forward.bw.mdsum
md5sum $OUTDIR/$BASE.reverse.bw > $MD5/$BASE.reverse.bw.mdsum
