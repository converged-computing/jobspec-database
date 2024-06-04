#!/bin/bash
#PBS -N deepconsensus
#PBS -P xl04
#PBS -q normal
#PBS -l storage=gdata/xl04+gdata/if89+gdata/te53
#PBS -l walltime=48:00:00
#PBS -l mem=2GB
#PBS -l ncpus=1
#PBS -l wd

module load nextflow
nextflow /g/data/xl04/ka6418/github/ausarg/scripts/deepconsensus.nf \
 --subreads ${subreads} \
 --output ${outdir} \
 --chunks ${chunks} \
 --sample ${sample} \
 --workdir $(dirname "$subreads") \
 -with-report "${outdir}/${sample}_report.html" \
 -with-trace "${outdir}/${sample}_trace.txt"  \
 -with-timeline "${outdir}/${sample}_timeline.html" \
 -with-dag "${outdir}/${sample}_graph.dot" \
 -resume