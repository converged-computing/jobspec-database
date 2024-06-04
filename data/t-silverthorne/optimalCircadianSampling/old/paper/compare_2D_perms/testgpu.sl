#!/bin/bash -l
#SBATCH --job-name=testperms
#SBATCH --account=def-stinch   
#SBATCH --time=0:02:00          # adjust this to match the walltime of your job

#SBATCH --nodes=1      
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12      # adjust this if you are using parallel commands
#SBATCH --mem=2000             # adjust this according to the memory requirement per node you need

#SBATCH --mail-user=turner.silverthorne@utoronto.ca 
#SBATCH --mail-type=ALL

# Choose a version of MATLAB by loading a module:
module load matlab/2022b.2
matlab -nodisplay -r "testgpu"
