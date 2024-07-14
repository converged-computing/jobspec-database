#!/usr/bin/bash
#SBATCH -N 1 -n 24 --mem 64gb --out logs/make_cov.%a.log -p short

hostname # for debugging and cleanup
module load bwa
module load samtools/1.11
module load bedtools

N=${SLURM_ARRAY_TASK_ID}
CPU=1
if [ $SLURM_CPUS_ON_NODE ]; then
 CPU=$SLURM_CPUS_ON_NODE
fi


if [ -z $N ]; then
 N=$1
fi

if [ -z $N ]; then
 echo "need to provide a number by --array or cmdline"
 exit
fi
ASM=results/genome_asm
OUT=results/coverage
mkdir -p $OUT
INFOLDER=input
SAMPLES=samples_prefix.csv
IFS=,
tail -n +2 $SAMPLES | sed -n ${N}p | while read SPECIES STRAIN JGILIBRARY BIOSAMPLE BIOPROJECT TAXONOMY_ID ORGANISM_NAME SRA_SAMPID SRA_RUNID LOCUSTAG TEMPLATE
do
	echo -n "Start Time "
	date
    # Determine output directory
    STEM=$(echo -n $SPECIES | perl -p -e 's/\s+/_/g')
    ASMFILE=$ASM/${STEM}.merge.fa
    if [ ! -f $ASMFILE ]; then
	echo "no merged $ASMFILE"
	break
    fi
    FWD=${INFOLDER}/${STEM}_R1.fq.gz
    REV=${INFOLDER}/${STEM}_R2.fq.gz
    
    echo "Processing $STEM"
    # small speedup would be to write this to /scratch instead of the current directory
    SAM=/scratch/$STEM.remap.sam
    BAM=/scratch/$STEM.remap.bam
    COV=$OUT/$STEM.cov
    COVBED=$OUT/$STEM.genome_cov.bed
    COVTAB=$OUT/$STEM.coverage.tab
    if [ ! -s $COVTAB ]; then
	if [ ! -s $ASMFILE.bwt ]; then
	    bwa index $ASMFILE
	fi
	if [ ! -s $ASMFILE.fai ]; then
	    samtools faidx $ASMFILE
	fi
	bwa mem -t $CPU $ASMFILE $FWD $REV > $SAM
	samtools sort --threads $CPU -T /scratch -O bam -o $BAM $SAM
	samtools index $BAM
	
	# can replace this also with samtools faidx and a cut cmd
	#fasta_length_table.pl $ASSEMBLY > $BASE.genome.lengths
	genomeCoverageBed -ibam $BAM  > $COVBED
	
	module load autometa/1.0.2
	source activate autometa
	
	contig_coverage_from_bedtools.pl $COVBED > $COVTAB
#	make_contig_table.py -a $ASMFILE -c $COVTAB -o $COV
	rm -f $SAM $BAM $BAM.bai
    fi
echo -n "End Time "
date
done
