#!/bin/bash
#PBS -l select=1:ncpus=4
#PBS -N stemness_brca
#PBS -q ct160
#PBS -P ACD110174
#PBS -J 1-10
#PBS -j oe
#PBS -M r08942073@ntu.edu.tw
#PBS -m be

BioMarkers=('' 'ERBB2' 'ESR1' 'MKI67' 'PGR' 'PLAU')

cd $PBS_O_WORKDIR

module purge

echo "${BioMarkers[${PBS_ARRAY_INDEX}]}"

./source/stemness.sh ${BioMarkers[${PBS_ARRAY_INDEX}]} 'TCGA-BRCA'