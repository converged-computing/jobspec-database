#!/bin/bash
#SBATCH --partition=general

#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=60000
#SBATCH --time=100:00:00
#SBATCH --mail-user=jagath@caltech.edu
#SBATCH --mail-type=ALL

module load Python
module load matplotlib

python roc_prc_gen.py [input file] [output file] 
