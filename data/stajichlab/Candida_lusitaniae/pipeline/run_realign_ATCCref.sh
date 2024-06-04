#PBS -l nodes=1:ppn=1,mem=16gb,walltime=25:00:00 -q js -j oe -N realign
module load java
module load GATK
module load picard

MEM=16g
GENOMEIDX=/shared/stajichlab/projects/HMAC/Clus_reseq/Aln/ATCC_Ref/candida_lusitaniae_ATCC_42720.fasta
BAMDIR=bam_ATCC
SAMPLEFILE=samples.info
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

ROW=`head -n $LINE $SAMPLEFILE | tail -n 1 | awk '{print $1}'`

echo "ROW is $ROW"
if [ ! -f $BAMDIR/$ROW.DD.bam ]; then
  echo "Error expected a Duplication Marked bam file ($ROW.DD.bam)"
  exit
fi
if [ ! -f $BAMDIR/$ROW.DD.bai ]; then
 java -jar $PICARD/BuildBamIndex.jar I=$BAMDIR/$ROW.DD.bam
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
