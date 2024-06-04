#!/bin/bash

#SBATCH --mail-user=nick@nickeubank.com  # email address
#SBATCH --mail-type=ALL  # Alerts sent when job begins, ends, or aborts
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10G
#SBATCH --job-name=fully
#SBATCH --array=1-250
#SBATCH --time=02-00:00:00  # Wall Clock time (dd-hh:mm:ss) [max of 14 days]
#SBATCH --output=outputs/fully_%A_%a.output  # output and error messages go to this file

export CHUNK_SIZE=4

julia fully_arrayed_individual_sim.jl
