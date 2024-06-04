#!/bin/bash

#SBATCH -q normal
#SBATCH -N 4
#SBATCH -t 08:00:00
#SBATCH -J  policy_real
#SBATCH --mail-user=nkozak@stanford.edu
#SBATCH --mail-type=ALL


module purge
module load openmpi
module load anaconda3 
module load julia 
eval "$(conda shell.bash hook)"
conda activate env_CS238
cd /home/nkozak/CS238/explore_eps
python3 p_ep0.1.py 
