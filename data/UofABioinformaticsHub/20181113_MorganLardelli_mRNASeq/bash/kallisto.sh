#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=01:00:00
#SBATCH --mem=4GB
#SBATCH -o /data/biohub/20181113_MorganLardelli_mRNASeq/slurm/%x_%j.out
#SBATCH -e /data/biohub/20181113_MorganLardelli_mRNASeq/slurm/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stephen.pederson@adelaide.edu.au

# Load modules
module load kallisto/0.43.1-foss-2016b
module load SAMtools/1.3.1-foss-2016b

## Reference Files
IDX=/data/biorefs/reference_genomes/ensembl-release-94/danio-rerio/kallisto/Danio_rerio.GRCz11.cdna.inc.psen2mutants

## Directories
PROJROOT=/data/biohub/20181113_MorganLardelli_mRNASeq
TRIMDATA=${PROJROOT}/1_trimmedData

## Setup for kallisto output
ALIGNDATA=${PROJROOT}/3_kallisto

## Now organise the input files
F1=$1
F2=${F1%R1.fq.gz}2_R1.fq.gz

## Organise the output files
OUTDIR=${ALIGNDATA}/$(basename ${F1%_R1.fq.gz})
echo -e "Creating ${OUTDIR}"
mkdir -p ${OUTDIR}
OUTBAM=${OUTDIR}/$(basename ${F1%_R1.fq.gz}.bam)

echo -e "Currently aligning:\n\t${F1}\n\t${F2}"
echo -e "Output will be written to ${OUTDIR}"
kallisto quant \
	-b 50 \
	--pseudobam \
	--single \
	--fr-stranded \
	-l 200 \
	-s 20 \
	-i ${IDX} \
	-o ${OUTDIR} \
	${F1} ${F2} | \
	samtools view -F4 -bS - > ${OUTBAM}

