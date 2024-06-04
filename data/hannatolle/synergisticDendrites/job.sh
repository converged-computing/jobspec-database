#!/bin/bash

#PBS -l select=1:ncpus=8:mem=8gb
#PBS -l walltime=08:00:00
#PBS -N dend_sims
#PBS -J 1-2
 
module load anaconda3/personal
source activate py311

cd ~/projects/synergisticDendrites/
python main.py ${HOME} ${PBS_ARRAY_INDEX}
