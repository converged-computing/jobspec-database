#!/bin/sh
#SBATCH --chdir=./
#SBATCH --job-name=step2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=quanah
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=1-46

module load singularity

name_array=$( head -n${SLURM_ARRAY_TASK_ID} helper3.txt | tail -n1 )

interval_array=$( head -n${SLURM_ARRAY_TASK_ID} helper3b.txt | tail -n1 )

export SINGULARITY_CACHEDIR="/lustre/work/johruska/singularity-cachedir"

singularity exec $SINGULARITY_CACHEDIR/gatk_4.2.3.0.sif gatk --java-options "-Xmx58g" GenomicsDBImport --genomicsdb-shared-posixfs-optimizations -V /lustre/scratch/johruska/central_america_pine_oak/melanerpes/02_vcf/${name_array}._Melanerpes_formicivorus_KU_8974_CHAL_.g.vcf -V /lustre/scratch/johruska/central_america_pine_oak/melanerpes/02_vcf/${name_array}._Melanerpes_formicivorus_KU_8976_CHAL_.g.vcf -V /lustre/scratch/johruska/central_america_pine_oak/melanerpes/02_vcf/${name_array}._Melanerpes_formicivorus_KU_33511_NSG_.g.vcf -V /lustre/scratch/johruska/central_america_pine_oak/melanerpes/02_vcf/${name_array}._Melanerpes_formicivorus_KU_33599_ALA_.g.vcf -V /lustre/scratch/johruska/central_america_pine_oak/melanerpes/02_vcf/${name_array}._Melanerpes_formicivorus_KU_33614_ALA_.g.vcf -V /lustre/scratch/johruska/central_america_pine_oak/melanerpes/02_vcf/${name_array}._Colaptes_mexicanoides_KU_132431_COL_.g.vcf --genomicsdb-workspace-path /lustre/scratch/johruska/central_america_pine_oak/melanerpes/02_vcf/${name_array} -L ${interval_array}
