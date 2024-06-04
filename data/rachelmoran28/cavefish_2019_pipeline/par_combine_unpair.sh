#!/bin/bash
#PBS -l nodes=1:ppn=1,walltime=2:00:00 
#PBS -m abe 
#PBS -j oe
#PBS -M rmoran@umn.edu 
#PBS -q mesabi
#PBS -N par_combine_unpaired_Jineo


#This script is used to combine unpaired reads from R1 and R2 fastq files prior to alignment with bwa


#       The home folder
export HOME=/home/mcgaughs/shared/Datasets/Reads_ready_to_align/Jineo

module load parallel

#       Define a bash function for doing the processing

parcomb() {
        #sample basename from the <POP>_SampID.txt file supplied to parallel with the -a flag
        SAMP=${1}
        cat ${SAMP}_adtrim_trim_unpair_R1.fastq.gz ${SAMP}_adtrim_trim_unpair_R2.fastq.gz > ${SAMP}_adtrim_trim_unpair_all.fastq.gz
}

#   Export function so we can call it with parallel
export -f parcomb
#       cd into the reads directory
cd ${HOME}

parallel --joblog Jineo_combunpair_parallel_logfile.txt -a Jineo_SampID.txt parcomb
