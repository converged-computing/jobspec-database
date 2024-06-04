#!/bin/bash 
#SBATCH --ntasks=40                                                            
#SBATCH --ntasks-per-core=1
#SBATCH -J spatialR
#SBATCH -o %x-%j.o
#SBATCH -e %x-%j.e
#SBATCH -t 0-03:59:59
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mvanega1@asu.edu

module purge
module load julia/1.5.0

export SLURM_NODEFILE=`generate_pbs_nodefile`
julia --machine-file $SLURM_NODEFILE ~/SpatialRust/scripts/ParamScan5.jl

cp /scratch/mvanega1/track05/* ~/SpatialRust/results/track05/
rm /scratch/mvanega1/track05/*
