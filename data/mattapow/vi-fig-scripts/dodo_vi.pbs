#!/bin/bash

# This is a sample pbs job to run Dodonaphy's in 
# variational inference mode

#PBS -N VI
#PBS -l ncpus=1
#PBS -l mem=200kb
#PBS -l walltime=10:00:00
#PBS -e vi.e
#PBS -o vi.o
#PBS -J 1-8

cd ${PBS_O_WORKDIR}
source miniconda3/bin/activate
source activate dodonaphy

ds=${PBS_ARRAY_INDEX}
dim=20
crv=-1000


dodo --infer vi \
--connect nj \
--embed up \
--path_root analysis/ds${ds} \
--path_dna data/DS.nex \
--prior None \
--epochs 1000 \
--draws 1000 \
--start mb/sumt/DS.nex.con.tre \
--dim ${dim} \
--curv ${crv} \
--learn 0.01 \
--temp 0.00001 \
--importance 2 \
--boosts 1 \

conda deactivate
