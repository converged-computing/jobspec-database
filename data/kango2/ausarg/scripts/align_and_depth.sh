#!/bin/bash
#PBS -N align_and_depth
#PBS -P xl04
#PBS -q normal
#PBS -l storage=gdata/xl04+gdata/if89
#PBS -l walltime=48:00:00
#PBS -l mem=2GB
#PBS -l ncpus=1
#PBS -l wd

module load nextflow
nextflow run /g/data/xl04/ka6418/github/ausarg/scripts/align_and_depth.nf \
 --pbFiles ${pb} \
 --ontFiles ${ont} \
 --outdir ${outdir} \
 --ref ${ref} \
 --sample ${sample} \
 -with-report "${outdir}/${sample}_report.html" \
 -with-trace "${outdir}/${sample}_trace.txt"  \
 -with-timeline "${outdir}/${sample}_timeline.html" \
 -with-dag "${outdir}/${sample}_graph.dot"