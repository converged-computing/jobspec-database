#!/bin/bash

#SBATCH --account=hpc_p_anderson
#SBATCH --constraint=skylake
#SBATCH --job-name=nf
#SBATCH --cpus-per-task=40
#SBATCH --mem=185G
#SBATCH --time=20:00:00
#SBATCH --output=/globalhome/hxo752/HPC/slurm_logs/%j.out

module --force purge
module load StdEnv/2020
module load nextflow/22.04.3
module load gentoo/2020
module load singularity/3.9.2

OUTDIR="/globalhome/hxo752/HPC/ngsf_git_repos/NGSF-core-projects/DNAMethylation"
DATA="/datastore/NGSF001/datasets/bisulfite_seq/"
#config_file=$1;
condition="Colon_Normal_Primary"

mkdir -p  ${OUTDIR}/analysis
mkdir -p  ${OUTDIR}/analysis/results
mkdir -p  ${OUTDIR}/analysis/work
#echo ${SLURM_TMPDIR}


#nextflow run nf-core/methylseq -profile singularity -c testdata.config
nextflow run nf-core/methylseq -profile singularity \
                               --input ${DATA}/${condition}/'*_{1,2}.fastq.gz' \
                               -w ${OUTDIR}/analysis/work \
                               --outdir ${OUTDIR}/analysis/results \
                               --genome GRCh38 \
                               --single_end false
                               
#rsync -rvzP  ${SLURM_TMPDIR}/chip_results ${OUTDIR}
