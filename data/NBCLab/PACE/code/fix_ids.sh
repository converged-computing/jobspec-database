#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --job-name=fixids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1gb
#SBATCH --partition=bluemoon
# Outputs ----------------------------------
#SBATCH --output=log/%x_%j.out   
#SBATCH --error=log/%x_%j.err   
# ------------------------------------------

pwd; hostname; date
set -e


#==============Shell script==============#
#Load the software needed
# module load python/python-miniconda3-rdchem-deepchem
spack load python@3.7.7
# source activate /gpfs1/home/m/r/mriedel/pace/env/env_bidsify

python /gpfs1/home/m/r/mriedel/pace/dsets/code/fix_ids.py


date
