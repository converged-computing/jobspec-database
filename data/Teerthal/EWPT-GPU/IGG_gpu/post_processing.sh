#!/bin/bash

#SBATCH -c 26
#SBATCH -t 0-04:00                  # wall time (D-HH:MM)
#SBATCH --mem=0
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)

time julia -t 26 post_sim.jl