#!/bin/bash
#SBATCH --job-name="jobname"
#SBATCH -A p31412
#SBATCH -p long    ## partition
#SBATCH -N 1  ## number of nodes
#SBATCH --ntasks-per-node=64  ## number of cores
#SBATCH -t 168:00:00
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

#https://researchcomputing.princeton.edu/support/knowledge-base/namd
module purge all
module load namd

/software/NAMD/2.13/verbs/charmrun /software/NAMD/2.13/verbs/namd2 +p$SLURM_NPROCS run.namd > run_namd.log
