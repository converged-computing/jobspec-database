#!/bin/bash

#SBATCH --job-name=1000_runs
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=um21226@bris.ac.uk

#SBATCH --mem=25000
#SBATCH --account=cosc029884

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --time=12:00:00

#SBATCH --partition=cpu
#SBATCH --array=1-1000

#SBATCH --output=/user/home/um21226/out_directory/density_2.6__alpha_10/%a_density_2.6__alpha_10.out

module purge

. ~/initConda.sh

conda activate diss

python -u ./code/main.py -density 2.3 -alpha 0
