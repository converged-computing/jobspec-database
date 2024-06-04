#!/bin/sh
#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=4:mem=12gb
#PBS -N NF_STAR_Coordinator
#PBS -j oe

Project_Dir=/path/to/project/dir
cd $Project_Dir

module load nextflow/22.04.4
module load java/jdk-16

echo "Starting: `date`"
nextflow run STAR_Align.nf \
	-c STAR_Align.config \
	--profile imperial \
	--Mode "PE" \
	--RefGen "/path/to/assembly.fasta" \
	--RefGTF "/path/to/assembly.gtf" \
	--InDir "/path/to/raw/reads" && echo "Finished: `date`"