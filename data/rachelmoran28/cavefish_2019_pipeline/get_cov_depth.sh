#!/bin/bash
#PBS -l mem=22gb,nodes=1:ppn=12,walltime=24:00:00 
#PBS -m abe 
#PBS -j oe
#PBS -M rmoran@umn.edu 
#PBS -q mesabi
#PBS -N bamg_cov_depth

module load parallel/20160822
module load samtools/1.7

#Can use any bam aligned to the cavefish genome for this step
export CAVE_GENOME_SIZE=$(samtools view -H /home/mcgaughs/shared/Datasets/bams/cave.fish.reference.genome/v1_cavefish_raw_bams/Jineo_1_dupsmarked_rgadd.bam | grep -P '^@SQ' | cut -f 3 -d ':' | awk '{sum+=$1} END {print sum}')
export HOME=/home/mcgaughs/shared/Datasets/bams/cave.fish.reference.genome/v1_cavefish_raw_bams

#Can use any bam aligned to the surface fish genome for this step
#export SURFACE_GENOME_SIZE=$(samtools view -H /home/mcgaughs/shared/Datasets/bams/surface.fish.reference.genome/v2_surfacefish_raw_bams/Jineo_1_dupsmarked_rgadd.bam | grep -P '^@SQ' | cut -f 3 -d ':' | awk '{sum+=$1} END {print sum}')
#export HOME=/home/mcgaughs/shared/Datasets/bams/surface.fish.reference.genome/v2_surfacefish_raw_bams


bam_depth() {
    BAM=${1}
    BASE=$(basename $BAM)
    ONAME=${BASE/.bam/}
    samtools depth -aa ${BAM} | awk -v name=$ONAME -v size=$CAVE_GENOME_SIZE '{sum+=$3} END {print "Cave\t"name"\t"sum/size}' >> ${HOME}/cave_genome_depths.txt
}

export -f bam_depth

echo -n "Ran: "
date

find /home/mcgaughs/shared/Datasets/bams/cave.fish.reference.genome/v1_cavefish_raw_bams -type f -name *.bam | parallel bam_depth > ${HOME}/cave_genome_depths.txt
#find /home/mcgaughs/shared/Datasets/bams/surface.fish.reference.genome/v2_surfacefish_raw_bams -type f -name *.bam | parallel bam_depth > ${HOME}/surface_genome_depths.txt

echo -n "Done: "
date
