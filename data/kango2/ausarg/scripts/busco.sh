#!/bin/bash

#PBS -N BUSCO
#PBS -P xl04
#PBS -q normalsr
#PBS -l walltime=20:00:00
#PBS -l mem=512GB
#PBS -l ncpus=104
#PBS -j oe
#PBS -l storage=gdata/xl04+gdata/if89
#PBS -l wd
#PBS -l jobfs=100GB
#PBS -M kirat.alreja@anu.edu.au


module load singularity

#inputs : fasta, outpath

lineage=/g/data/xl04/bpadata/Bassiana_duperreyi/projects/chromsyn/busco_downloads
prefix=$(basename "${fasta}" .fasta)
cd ${outpath}

singularity exec /g/data/xl04/ka6418/docker_images/busco-5.4.7.sif busco -o run_$prefix --offline -i $fasta -l sauropsida_odb10 --download_path $lineage --cpu ${PBS_NCPUS} -m genome --tar