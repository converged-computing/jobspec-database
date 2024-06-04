#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1
#SBATCH --output=slurm/real/%A_%a.out

# load the required modules
module load r
module load gcc/11.2.0

# singularity run -B /scratch,/m,/l,/share /scratch/cs/bayes_ave/stan-triton.sif Rscript ./R/real-world/ionosphere.R $1 $2
srun Rscript ./R/real-world/ionosphere.R $1 $2
