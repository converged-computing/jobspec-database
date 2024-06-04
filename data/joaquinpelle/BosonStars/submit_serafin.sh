#!/bin/bash -l
### Standard output and error:
#SBATCH -o logs/out.%j
#SBATCH -e logs/err.%j

### Initial working directory:
#SBATCH -D ./

### Job Name:
#SBATCH -J bosonstar

#SBATCH --partition=multi

### Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<user-email>
#
# Wall clock Limit:
#SBATCH --time=24:00:00

##################################

export JULIA_NUM_THREADS=64
export SLURM_HINT=multithread 

julia --project=BosonStars main.jl
