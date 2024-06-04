#PBS -l nodes=1:ppn=1,mem=32gb,walltime=25:00:00  -j oe -N realign
module unload java
module load java/8
module load gatk/3.6
module load picard

MEM=32g
GENOMEIDX=index/A_fumigatus_Af293.fasta
BAMDIR=aln
SAMPLEFILE=samples.csv
b=`basename $GENOMEIDX .fasta`
dir=`dirname $GENOMEIDX`
if [ ! -f $dir/$b.dict ]; then
 java -jar $PICARD CreateSequenceDictionary R=$GENOMEIDX O=$dir/$b.dict SPECIES="Aspergillus_fumigatus" TRUNCATE_NAMES_AT_WHITESPACE=true
fi

N=$PBS_ARRAYID
CPU=1
if [ $PBS_NP ]; then 
 CPU=$PBS_NP
fi

SAMPFILE=samples.csv
if [ ! $N ]; then
 N=$1
fi

if [ ! $N ]; then 
 echo "need to provide a number by PBS_ARRAYID or cmdline"
 exit
fi

if [ $N -eq "1" ]; then 
 echo "Skipping 1 as it is the header in the infile"
 exit
fi
MAX=`wc -l $SAMPFILE | awk '{print $1}'`

if [ $N -gt $MAX ]; then 
 echo "$N is too big, only $MAX lines in $SAMPFILE"
 exit
fi

IFS=,
sed -n ${N}p $SAMPFILE | while read SAMPLE STRAIN FWD REV BARCODE DESC;
do
if [ $REV ]; then
 SAMPLE=$SAMPLE.PE
else
 SAMPLE=$SAMPLE.SE
fi

hostname
echo "SAMPLE=$SAMPLE"
if [ ! -f $BAMDIR/$SAMPLE.DD.bam ]; then
   java -jar $PICARD MarkDuplicates I=$BAMDIR/$SAMPLE.RG.bam O=$BAMDIR/$SAMPLE.DD.bam METRICS_FILE=$SAMPLE.dedup.metrics CREATE_INDEX=true VALIDATION_STRINGENCY=SILENT
fi
echo "here"
if [ ! -f $BAMDIR/$SAMPLE.DD.bai ]; then
 java -jar $PICARD BuildBamIndex I=$BAMDIR/$SAMPLE.DD.bam TMP_DIR=/scratch
fi
echo "here"
if [ ! -f $BAMDIR/$SAMPLE.intervals ]; then 
 java -Xmx$MEM -jar $GATK \
   -T RealignerTargetCreator \
   -R $GENOMEIDX \
   -I $BAMDIR/$SAMPLE.DD.bam \
   -o $BAMDIR/$SAMPLE.intervals
fi
echo "here"
if [ ! -f $BAMDIR/$SAMPLE.realign.bam ]; then
 java -Xmx$MEM -jar $GATK \
   -T IndelRealigner \
   -R $GENOMEIDX \
   -I $BAMDIR/$SAMPLE.DD.bam \
   -targetIntervals $BAMDIR/$SAMPLE.intervals \
   -o $BAMDIR/$SAMPLE.realign.bam
fi
echo "done"

done
