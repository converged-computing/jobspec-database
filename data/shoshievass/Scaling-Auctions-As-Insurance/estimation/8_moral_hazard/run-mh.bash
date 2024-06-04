#!/bin/bash
#
#SBATCH --job-name=hazard
#
#SBATCH --time=2-0
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/zfs/projects/faculty/svass-bang/logs/mh-logs/%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --array=1-430

module load julia/1.7.3 gurobi
export JLGUROBI=true
julia --threads=4 --project 8-hazard.jl
