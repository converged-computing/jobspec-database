#!/bin/bash
#PBS -A dri-wasserman
#PBS -V
#PBS -N SetupSTR_Sim
#PBS -m bea
#PBS -M prichmond@cmmt.ubc.ca
#PBS -l select=1:ncpus=32:mem=100gb
#PBS -l walltime=240:0:0


## Working directory variables
STR_ANALYSIS_DIR=/project/dri-wasserman/RICHMOND/STR_Analysis/
FASTADIR=$STR_ANALYSIS_DIR/STR_Simulation/str_fastas/GRCh38/
BAMDIR=$STR_ANALYSIS_DIR/STR_Simulation/str_bams/
WRITE_DIR=$BAMDIR/GRCh38/
WORKING_DIR=$WRITE_DIR

mkdir -p $WORKING_DIR

## Genome processing variables
BWA_INDEX=/project/dri-wasserman/GENOME/GRCh38/GRCh38-lite.fa
GENOME_FASTA=/project/dri-wasserman/GENOME/GRCh38/GRCh38-lite.fa
THREADS=3

## Fastas to simulate. 
Files=(${FASTADIR}*fasta)

#Varsim variables
ART=art_illumina
READLENGTH=150
DEPTH_HOMO=36
DEPTH_HET=18
DEPTHS=( $DEPTH_HOMO $DEPTH_HET )

# Calculated with samtools stats on the human genome BAM file from Polaris
FRAGMENT_LENGTH=460
FRAGMENT_STDEV=115

#####################


# Phase II: Simulate Reads From Fasta, map reads, and convert them within a loop

cd $WORKING_DIR
for fasta in $(ls ${FASTADIR}*fasta)
do
	echo $fasta
	IFS='/' read -a array <<< $fasta
	SampleFasta=${array[-1]}
	IFS='.' read -a array2 <<< "$SampleFasta"
	SAMPLE_ID_BASE=${array2[0]}
	echo $SAMPLE_ID_BASE
	
	for DEPTH in ${DEPTHS[@]}
	do
		echo $DEPTH
		SAMPLE_ID=${SAMPLE_ID_BASE}_${DEPTH}x
		echo $SAMPLE_ID
		$ART -l $READLENGTH \
			-f $DEPTH \
			-o $SAMPLE_ID. \
			-m $FRAGMENT_LENGTH \
			-s $FRAGMENT_STDEV \
			-i $fasta 
		
		### Phase III: Map the simulated reads, and convert to bam
		FASTQR1=$WRITE_DIR$SAMPLE_ID.1.fq
		FASTQR2=$WRITE_DIR$SAMPLE_ID.2.fq
		
		#Map with BWA
		bwa mem $BWA_INDEX -t $THREADS -R "@RG\tID:$SAMPLE_ID\tSM:$SAMPLE_ID\tPL:illumina" -M $FASTQR1 $FASTQR2 > $WORKING_DIR$SAMPLE_ID.sam
		samtools view -@ $THREADS -ubS $WORKING_DIR$SAMPLE_ID'.sam' \
			| samtools sort - -@ $THREADS  -T $WORKING_DIR$SAMPLE_ID'.sorted' -O BAM -o $WORKING_DIR$SAMPLE_ID'.sorted.bam'
		samtools index $WORKING_DIR$SAMPLE_ID'.sorted.bam'
		
	done

done
