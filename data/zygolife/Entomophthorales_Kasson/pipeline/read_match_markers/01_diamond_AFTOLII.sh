#!/usr/bin/bash -l
#SBATCH -p short -N 1 -c 96 -n 1 --mem 24gb --out logs/diamond_blastx.%a.log -a 1-14

module load diamond
module load workspace/scratch

CPU=$SLURM_CPUS_ON_NODE
if [ -z $CPU ]; then
	CPU=1
fi
N=${SLURM_ARRAY_TASK_ID}

if [ -z $N ]; then
    N=$1
    if [ -z $N ]; then
        echo "Need an array id or cmdline val for the job"
        exit
    fi
fi

DB=AFTOL_search/AFTOL_II
OUT=AFTOL_search/search
mkdir -p $OUT
SAMPLEFILE=samples.csv
FASTQ=input
IFS=, # set the delimiter to be ,
tail -n +2 $SAMPLEFILE | sed -n ${N}p | while read BASE ILLUMINASAMPLE SPECIES STRAIN PROJECT DESCRIPTION ASMFOCUS
do
    LEFTIN=$FASTQ/${BASE}_${ILLUMINASAMPLE}_R1_001.fastq.gz
    RIGHTIN=$FASTQ/${BASE}_${ILLUMINASAMPLE}_R2_001.fastq.gz
    echo " running diamdon of $LEFTIN against $DB"
    if [ ! -s $OUT/$STRAIN.R1.AFTOL_II.aln ]; then
      diamond blastx --tmpdir $SCRATCH -p $CPU --db $DB -q $LEFTIN --iterate --ultra-sensitive --out $OUT/$STRAIN.R1.AFTOL_II.aln
    fi
    if [ ! -s $OUT/$STRAIN.R2.AFTOL_II.aln ]; then
      diamond blastx --tmpdir $SCRATCH -p $CPU --db $DB -q $RIGHTIN --iterate --ultra-sensitive --out $OUT/$STRAIN.R2.AFTOL_II.aln
    fi
    
done
