#!/bin/bash

#SBATCH -J regnn57
#SBATCH -p general
#SBATCH -o regnn57_%j.txt
#SBATCH -e regnn57_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mraina@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=9:00:00
#SBATCH --mem=200G
#SBATCH -A r00206


#Load any modules that your program needs
module load miniconda

#Activate Conda enviorment
source activate /N/slate/mraina/egnn/

#Change Directories
cd /N/slate/mraina/REGNN/

#Run your program
python calculate_ARI.py


