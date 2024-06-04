#!/usr/bin/bash
#SBATCH -p intel -N 1 -n 16 --mem 32gb --out logs/make_gvcf.%a.log --time 48:00:00

module load picard
module load java/13
module load gatk/4
module load bcftools

MEM=32g
ALNFOLDER=aln
VARIANTFOLDER=gvcf
HTCFORMAT=cram #default but may switch back to bam
HTCFOLDER=cram # default
HTCEXT=cram
SAMPINFO=samples.csv
CENTER=UCR

if [ -f config.txt ]; then
    source config.txt
fi

DICT=$(echo $REFGENOME | sed 's/fasta$/dict/')

if [ ! -f $DICT ]; then
	picard CreateSequenceDictionary R=$GENOMEIDX O=$DICT
fi
mkdir -p $VARIANTFOLDER
TEMP=/scratch
CPU=1
if [ $SLURM_CPUS_ON_NODE ]; then
 CPU=$SLURM_CPUS_ON_NODE
fi
N=${SLURM_ARRAY_TASK_ID}

if [ ! $N ]; then
 N=$1
fi

if [ ! $N ]; then
 echo "need to provide a number by --array slurm or on the cmdline"
 exit
fi

hostname
date
IFS=,
tail -n +2 $SAMPINFO | cut -d, -f2 | sort | uniq | sed -n ${N}p | while read STRAIN
do
  # BEGIN THIS PART IS PROJECT SPECIFIC LIKELY
  # END THIS PART IS PROJECT SPECIFIC LIKELY
  STRAIN=$(echo $STRAIN | perl -p -e 's/[\/ ]/_/g')
  echo "STRAIN is $STRAIN"
  
  GVCF=$VARIANTFOLDER/$STRAIN.g.vcf
  INPUT=""
  NEWALN=0
  for ALNFILE in $(ls $ALNFOLDER/$STRAIN.*.$HTCEXT)
  do
	if [ -z $INPUT ]; then
	  	INPUT="--input $ALNFILE"
	else 
		INPUT="$INPUT --input $ALNFILE"
	fi
	if [ $ALNFILE -nt $GVCF ]; then
		NEWALN=1
	fi
  done
  if [ -z $INPUT ]; then
	  echo "No Aligned file for $ALNFOLDER/$STRAIN.*.$HTCEXT"
	  exit
  fi
  echo "INPUT is $INPUT"
  if [ -s $GVCF.gz ]; then
    echo "Skipping $STRAIN - Already called $STRAIN.g.vcf.gz"
    exit
  fi
  if [[ ! -f $GVCF || $NEWALN -eq 1 ]]; then
      eval "time gatk --java-options -Xmx${MEM} HaplotypeCaller \
   	  --emit-ref-confidence GVCF --sample-ploidy 1 $INPUT --reference $REFGENOME \
   	  --output $GVCF --native-pair-hmm-threads $CPU \
	  -G StandardAnnotation -G AS_StandardAnnotation -G StandardHCAnnotation"
 fi
 bgzip --threads $CPU -f $GVCF
 tabix $GVCF.gz
done
date

