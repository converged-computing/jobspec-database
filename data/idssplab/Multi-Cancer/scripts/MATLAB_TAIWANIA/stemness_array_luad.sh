#!/bin/bash
#PBS -l select=1:ncpus=4
#PBS -N stemness_luad
#PBS -q cf160
#PBS -P ACD110174
#PBS -J 1-10
#PBS -j oe
#PBS -M r08942073@ntu.edu.tw
#PBS -m be

BioMarkers=('' 'ALCAM' 'CADM1' 'EPCAM' 'HIF1A' 'PKM' 'PTK7' 'SLC2A1')

cd $PBS_O_WORKDIR

module purge

echo "${BioMarkers[${PBS_ARRAY_INDEX}]}"

./source/stemness.sh ${BioMarkers[${PBS_ARRAY_INDEX}]} 'TCGA-LUAD'