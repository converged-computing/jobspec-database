#!/bin/bash 
#SBATCH --ntasks=450                                                         
#SBATCH --ntasks-per-core=1
#SBATCH -J ABC
#SBATCH -o %x-%j.o
#SBATCH -e %x-%j.e
#SBATCH -t 0-04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mvanega1@asu.edu

module purge
module load julia/1.5.0

export SLURM_NODEFILE=`generate_pbs_nodefile`
julia --machine-file $SLURM_NODEFILE ~/SpatialRust/scripts/Agave/ABCinit.jl parameters.csv
