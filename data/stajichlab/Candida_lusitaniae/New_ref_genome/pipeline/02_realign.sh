#PBS -l nodes=1:ppn=1,mem=24gb,walltime=25:00:00  -j oe -N realign
module load java
module load gatk
module load picard

MEM=24g
GENOMEIDX=/bigdata/stajichlab/shared/projects/Candida/HMAC/Clus_reseq/New_ref_assembly/ref/S9.fasta
BAMDIR=bam
SAMPLEFILE=samples.info
b=`basename $GENOMEIDX .fasta`
dir=`dirname $GENOMEIDX`
if [ ! -f $dir/$b.dict ]; then
 java -jar $PICARD CreateSequenceDictionary R=$GENOMEIDX O=$dir/$b.dict SPECIES="Candida lusitaniae" TRUNCATE_NAMES_AT_WHITESPACE=true
fi

if [ ! $CPU ]; then
 CPU=1
fi

LINE=$PBS_ARRAYID

if [ ! $LINE ]; then
 LINE=$1
fi

if [ ! $LINE ]; then
 echo "Need a number via PBS_ARRAYID or cmdline"
 exit
fi
ROW=$(sed -n ${LINE}p $SAMPLEFILE | awk '{print $1}')

echo "ROW is $ROW"
if [ ! -f $BAMDIR/$ROW.DD.bam ]; then
  echo "Error expected a Duplication Marked bam file ($ROW.DD.bam)"
  exit
fi
if [ ! -f $BAMDIR/$ROW.DD.bai ]; then
 java -jar $PICARD BuildBamIndex I=$BAMDIR/$ROW.DD.bam
fi

if [ ! -f $BAMDIR/$ROW.intervals ]; then 
 java -Xmx$MEM -jar $GATK \
   -T RealignerTargetCreator \
   -R $GENOMEIDX \
   -I $BAMDIR/$ROW.DD.bam \
   -o $BAMDIR/$ROW.intervals
fi

if [ ! -f $BAMDIR/$ROW.realign.bam ]; then
 java -Xmx$MEM -jar $GATK \
   -T IndelRealigner \
   -R $GENOMEIDX \
   -I $BAMDIR/$ROW.DD.bam \
   -targetIntervals $BAMDIR/$ROW.intervals \
   -o $BAMDIR/$ROW.realign.bam
fi
