#!/bin/bash
#PBS -l nodes=1:ppn=1,mem=126gb,walltime=96:00:00
#PBS -j oe
#PBS -q mangi
#PBS -N RmvIndels
#PBS -M rmoran@umn.edu 
#PBS -m abe
#PBS -t 1-35

module load bcftools
export PATH=$PATH:/home/mcgaughs/rmoran/software/tabix-0.2.6

# We have defined a file-of-files that lists regions to operate over. This is
# because we separated variant calling by chromosome. Chromosomes 1-25
# have their own files and the unmapped contigs get their own files (broken up into Un 1-10)
REGFOF="/home/mcgaughs/shared/References/2017-09_Astyanax_mexicanus/GATK_Regions_final.fof"
REGION=$(sed "${PBS_ARRAYID}q;d" ${REGFOF})
# We define the output region based on the basename of the intervals file that
# was read from the file-of-files
REG_BNAME=$(basename ${REGION})
REG_OUT=${REG_BNAME/.intervals/}

cd /home/mcgaughs/shared/Datasets/all_sites_LARGE_vcfs/filtered_surfacefish/combined_filtered/246Indvs_RepsMasked

module load python2

gunzip ${REG_OUT}_HardFiltered_246Indv_SurfaceRef_wInvar.RepsRemoved.vcf.gz

python2 Remove_indels_for_VCFtools.py ${REG_OUT}_HardFiltered_246Indv_SurfaceRef_wInvar.RepsRemoved.vcf > Indels_to_remove/${REG_OUT}_indels_to_remove.bed

bgzip ${REG_OUT}_HardFiltered_246Indv_SurfaceRef_wInvar.RepsRemoved.vcf
