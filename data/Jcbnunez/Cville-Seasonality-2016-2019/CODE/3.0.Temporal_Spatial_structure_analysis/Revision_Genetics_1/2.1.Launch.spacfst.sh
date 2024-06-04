#!/usr/bin/env bash
#
#SBATCH -J spacfst # A single job name for the array
#SBATCH -c 40
#SBATCH -N 1 # on one node
#SBATCH -t 24:00:00 #<= this may depend on your resources
#SBATCH --mem 40G #<= this may depend on your resources
#SBATCH -o ./slurmOut/spacfst.%A_%a.out # Standard output
#SBATCH -e ./slurmOut/spacfst.%A_%a.err # Standard error
#SBATCH -p bluemoon
#SBATCH --array=1-999

#1435 
# 

### load modules
module load spack/spack-0.18.1
spack load r@4.2.1 r-sf
spack load openjdk@11.0.15_10

### run script
Rscript \
2.spatial.fst.stability.R \
${SLURM_ARRAY_TASK_ID} \
0

if [ ${SLURM_ARRAY_TASK_ID} -le 436 ]
then

echo "shift activated!"
Rscript \
2.spatial.fst.stability.R \
${SLURM_ARRAY_TASK_ID} \
999

fi

echo "done"
date
