#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=96:00:00
#SBATCH --partition=matgen_prior
#SBATCH --account=matgen
#SBATCH --job-name=snl_tasks_atomate
#SBATCH --output=snl_tasks_atomate-%j.out
#SBATCH --error=snl_tasks_atomate-%j.error

module unload python
module unload virtualenv
module load python/3.4-anaconda
#module unload intel
source activate /global/u1/h/huck/ph_atomate

srun -n 1 python snl_tasks_atomate.py
