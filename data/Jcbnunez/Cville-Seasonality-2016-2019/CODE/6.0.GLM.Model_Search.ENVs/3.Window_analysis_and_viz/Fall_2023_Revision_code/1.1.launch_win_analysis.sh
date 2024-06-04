#!/usr/bin/env bash
#
#SBATCH -J glm.win # A single job name for the array
#SBATCH -c 40
#SBATCH -N 1 # on one node
#SBATCH -t 6:00:00 #<= this may depend on your resources
#SBATCH --mem 50G #<= this may depend on your resources
#SBATCH -p bluemoon
#SBATCH -o ./slurmOutput/win.%A_%a.out # Standard output
#SBATCH -e ./slurmOutput/win.%A_%a.err # Standard error
#SBATCH --array=3-4

module load spack/spack-0.18.1
spack load r@4.2.1 r-sf

met=master.file.fst.txt

n="${SLURM_ARRAY_TASK_ID}"

echo $n

Rscript \
--vanilla \
1.0.explore_window_analyses.R $n

date
echo "done"