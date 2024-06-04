#!/bin/bash -l

#SBATCH --job-name=merge_fixrg
#SBATCH -o merge_fixrg-%j.out 
#SBATCH -p main 
#SBATCH -n 1  
#SBATCH -N 1  
#SBATCH --cpus-per-task=40 

enable_lmod
module load container_env samtools
module load container_env/0.1
module load container_env java
export SINGULARITY_BIND=/home/e1garcia

BAMDIR=$1

BAMPATTERN=*-merged.rad.RAW-2-2-RG.bam

all_samples=$(ls $BAMDIR/$BAMPATTERN | sed -e 's/-merged\.rad\.RAW-2-2-RG\.bam//' -e 's/.*\///g')
all_samples=($all_samples)

sample_name=${all_samples[${SLURM_ARRAY_TASK_ID}]}
echo ${sample_name}

cd ${BAMDIR}

java -jar /home/e1garcia/shotgun_PIRE/pire_cssl_data_processing/scripts/picard.jar AddOrReplaceReadGroups I=${sample_name}-merged.rad.RAW-2-2-RG.bam O=${sample_name}-merged.rgfix.rad.RAW-2-2-RG.bam RGID=${sample_name}-merged RGLB=mergedlibs RGPL=illumina RGPU=unit1 RGSM=${sample_name}-merged 

crun.samtools samtools index ${sample_name}-merged.rgfix.rad.RAW-2-2-RG.bam

rm ${sample_name}-merged.rad.RAW-2-2-RG.bam
rm ${sample_name}-merged.rad.RAW-2-2-RG.bam.bai
