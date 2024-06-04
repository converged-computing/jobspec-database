#!/bin/bash

#SBATCH	 -J	sculptor_mc_orbit
#SBATCH -n 1                #Number of cores
#SBATCH -A durham				#Account name
#SBATCH -t 2:00:00
#SBATCH -p cosma			        #Queue name e.g. cosma, cosma6, etc.
#SBATCH -o %J.out		#Standard output file

module unload gnu_comp
module load intel_comp/2019
module load intel_mpi/2019
module load fftw/2.1.5 
module load gsl/2.4
# module load gnu_comp/7.3.0 
module load hdf5 #/1.8.20 

set -x
rm -f out/*
mpirun -np $SLURM_NTASKS /cosma/home/durham/dc-boru1/Gadget-RAPHAMW/source/Gadget2 params.txt > log.out

julia calc_peris.jl
