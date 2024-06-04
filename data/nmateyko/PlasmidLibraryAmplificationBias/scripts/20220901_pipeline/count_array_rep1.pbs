#!/bin/bash

#PBS -l walltime=1:00:00,select=1:ncpus=1:mem=2gb
#PBS -J 1-5
#PBS -N count_rep1
#PBS -M nicholas.mateyko@ubc.ca
#PBS -m abe
#PBS -A st-cdeboer-1
#PBS -o output_^array_index^.txt
#PBS -e error_^array_index^.txt
 
################################################################################
 
cd $PBS_O_WORKDIR
source /arc/software/spack/opt/spack/linux-centos7-x86_64/gcc-5.4.0/miniconda3-4.6.14-f4hr756q34tvp7nsjn7hovq46fomaww6/etc/profile.d/conda.sh
conda activate plab

set -e
logDir="log"
dataDir="/arc/project/st-cdeboer-1/nick/PlasmidLibraryAmplificationBias/data/consensus"
idx=$PBS_ARRAY_INDEX
mkdir -p log 

if [ ! -e $logDir/$idx.count.done ]
then
        /arc/project/st-cdeboer-1/nick/PlasmidLibraryAmplificationBias/scripts/count_reads.py -i $dataDir/consensus.$idx.txt.gz -o counts.$idx -l log.$idx.txt -d 2006000 -t 5        
 
        touch $logDir/$idx.count.done
else
        echo Already performed counts of $idx
fi
