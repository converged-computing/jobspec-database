#!/usr/bin/bash
#SBATCH -J GATK.HTC --out logs/GATK_HTC_novogene.%a.log --ntasks 8 --nodes 1 --mem 16G

module unload java
module load java/8
module load gatk/3.8
module load picard
module load tabix

MEM=32g
GENOMEIDX=genome/Af293.fasta
BAMDIR=bam
OUTDIR=gvcf

mkdir -p $OUTDIR

b=$(basename $GENOMEIDX .fasta)
dir=$(dirname $GENOMEIDX)

N=${SLURM_ARRAY_TASK_ID}
CPU=1
if [ $SLURM_CPUS_ON_NODE ]; then
 CPU=$SLURM_CPUS_ON_NODE
fi

SAMPFILE=Novogene_samples.csv
if [ ! $N ]; then
 N=$1
fi

if [ ! $N ]; then 
 echo "need to provide a number by --array slurm or on the cmdline"
 exit
fi

MAX=$(wc -l $SAMPFILE | awk '{print $1}')

if [ $N -gt $MAX ]; then 
 echo "$N is too big, only $MAX lines in $SAMPFILE"
 exit
fi

IFS=,

tail -n +2 $SAMPFILE | sed -n ${N}p | while read STRAIN PREFIX LEFT RIGHT
do
 hostname
 SAMPLE=$STRAIN
 echo "SAMPLE=$SAMPLE"
 BAMFILE=$BAMDIR/$SAMPLE.bam
 if [ ! -e $BAMFILE ]; then
  echo "Cannot find $SAMPLE.bam in $BAMDIR"
  exit
 fi
 if [ ! -f $OUTDIR/$SAMPLE.g.vcf.gz ]; then
 if [ ! -f $OUTDIR/$SAMPLE.g.vcf ]; then
  java -Xmx${MEM} -jar $GATK \
  -T HaplotypeCaller \
  -ERC GVCF \
  -ploidy 1 \
  -I $BAMFILE -R $GENOMEIDX \
  -o $OUTDIR/$SAMPLE.g.vcf -nct $CPU
 fi
 bgzip $OUTDIR/$SAMPLE.g.vcf
 tabix $OUTDIR/$SAMPLE.g.vcf.gz
 fi
done
