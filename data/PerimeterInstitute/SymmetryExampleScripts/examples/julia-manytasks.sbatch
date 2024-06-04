#!/bin/bash
#SBATCH --partition debugq
#SBATCH --job-name=julia-manytasks
#SBATCH --nodes=4
#SBATCH --ntasks-per-core=1
#SBATCH --output=julia-manytasks.out
#SBATCH --time=0:10:00

set -euxo pipefail

module load slurm
module load julia

pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
date

# Create a file with the host names of all the nodes assigned to this
# job. List each host 40 ($SLURM_CPUS_ON_NODE) times.
machinefile=$(mktemp)
seq $SLURM_CPUS_ON_NODE |
    xargs -n 1 -I '{}' scontrol show hostnames |
    sort >$machinefile

julia --machine-file $machinefile julia-manytasks.jl

date
