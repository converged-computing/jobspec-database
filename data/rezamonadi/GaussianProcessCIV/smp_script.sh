#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=128gb
#SBATCH --job-name="parSampling"
#SBATCH --time=10:00:00
#SBATCH -p  intel
#SBATCH --output="parSampling.out"
#SBATCH --mail-user=rmona003@ucr.edu
#SBATCH --mail-type=ALL

# loading matlab module
module load matlab
# run matlab script
matlab -nodesktop -nosplash -r "parpool('local', 32); WSamplePar;"
echo "----"
hostname

