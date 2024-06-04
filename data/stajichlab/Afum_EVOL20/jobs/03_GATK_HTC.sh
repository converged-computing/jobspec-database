#PBS -l nodes=1:ppn=16,mem=32gb,walltime=18:00:00  -j oe -N HTC.GATK
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

OUTDIR=Variants
mkdir -p $OUTDIR
b=`basename $GENOME .fasta`

N=$BAMDIR/$SAMPLE.realign.bam

if [ ! -f $OUTDIR/$SAMPLE.g.vcf ]; then
java -Xmx${MEM} -jar $GATK \
  -T HaplotypeCaller \
  -ERC GVCF \
  -stand_emit_conf 10 -stand_call_conf 30 \
  -ploidy 1 \
  -I $N -R $GENOMEIDX \
  -o $OUTDIR/$SAMPLE.g.vcf -nct $CPU
fi
done
