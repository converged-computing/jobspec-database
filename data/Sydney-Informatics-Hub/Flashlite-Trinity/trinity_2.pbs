#! /bin/bash

#########################################################
# 
# Platform: UQ Flashlite 
# Description: 
#	Trinity de novo transcriptome assembly.
#	This PBS script is part of the Staged Trinity workflow
#	that executes Trinity de novo transcriptome
# 	assembly on University of Queensland's Flashlite HPC.
# Usage: 
#	Run this script when trinity_1.pbs has completed successfully.
#	Edit the script replacing everything in <> with your
#	inputs and options. For very large assemblies (e.g. global
# 	mammalian transcriptomes), increase memory by setting
# 	#PBS -l select=1:ncpus=24:mem=500gb and mem='500G'
#	Typically, <200Gb memory and <24hours walltime is expected
#	for a global mammalian assembly
# Singularity containers:
#	https://data.broadinstitute.org/Trinity/TRINITY_SINGULARITY/
#
# If you use this script towards a publication, please acknowledge the
# Sydney Informatics Hub (or co-authorship, where appropriate).
#
# Suggested acknowledgement:
# The authors acknowledge the scientific and technical assistance 
# <or e.g. bioinformatics assistance of > of Sydney Informatics Hub, the 
# University of Sydney and resources and services provided by the 
# University of Queensland.
# 
#########################################################

#PBS -A <account>
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=24:mem=500gb
#PBS -N Trinity_job2

set -e

module load singularity/3.5.0

mem='500G'

# Set trap
# EXIT runs on any exit, signalled or not.
finish(){
                part=trinity_2
		echo "$(date): Archiving ${sampleout} in ${TMPDIR} to ${outpath}/${sampleout}/${part}.tar"
		cd ${TMPDIR}
		mv ${sampleout} ${part}
		tar -cf "${outpath}/${sampleout}/${part}.tar" "${part}"
		rm -rf ${outpath}/${sampleout}/trinity_1.tar
                echo "$(date): Saved to ${outpath}/${sampleout}/${part}"
}
trap finish EXIT

## SET VARIABLES: REPLACE <>
sample=<sample_name>
outpath=<path_to_output>
simg=<path_to_container>
first=<path_to_fastq1>
second=<path_to_fastq2>

# Do not edit
ver=`echo ${simg} | perl -pe 's/^.*trinityrnaseq\.(v.*)\.simg/$1/g'`
sampleout=${sample}_trinity_${ver}
export LOCALOUT=${TMPDIR}/${sampleout}
mkdir -p ${LOCALOUT}
tar -xf ${outpath}/${sampleout}/trinity_1.tar -C ${LOCALOUT}
mv ${LOCALOUT}/trinity_1/* ${LOCALOUT}
rm -rf ${LOCALOUT}/trinity_1

# Run trinity, stop after inchworm, do not run chrysalis
singularity exec -B ${TMPDIR} ${simg} Trinity \
		--seqType fq \
                --max_memory ${mem} \
                --left ${first} \
                --right ${second} \
                --no_normalize_reads \
		--CPU ${NCPUS} \
                --output ${LOCALOUT} \
                --verbose \
		--no_distributed_trinity_exec

echo "$(date): Finished trinity 2"
