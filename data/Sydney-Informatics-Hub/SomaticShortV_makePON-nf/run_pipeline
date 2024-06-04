#!/bin/bash 

#PBS -P wz54
#PBS -N make_PoN
#PBS -l walltime=24:00:00
#PBS -l ncpus=48
#PBS -l mem=1024GB
#PBS -q hugemem
#PBS -W umask=022
#PBS -l wd
#PBS -l storage=scratch/er01+scratch/wz54

#module load nextflow/21.04.1
module load singularity
module load gatk/4.1.8.1
module load java/jdk-13.33

// I have used a local installtion of nextflow here (latest version).. 
// Instead of using a full path to a local installation, I will (first) test and then change this to running nextflow with a specific version
// e.g. NXF_VER=20.04.0 nextflow run hello


# Using the latest version of nextflow 
/scratch/wz54/npd561/installations/nextflow run main.nf \
	--outDir ./results \
	-resume \
	-with-report excecution_report.html \
	-with-timeline timeline_report.html \
	-with-dag results/pipeline_flowchart.pdf 
