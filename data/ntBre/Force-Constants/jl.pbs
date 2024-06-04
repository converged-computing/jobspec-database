#!/bin/sh
#PBS -N job
#PBS -S /bin/bash
#PBS -j oe
#PBS -W umask=022
#PBS -l walltime=00:30:00
#PBS -l ncpus=1
#PBS -l mem=32gb

module load julia/1.1.0
cd ~/research/Force-Constants/.
/ddn/home1/r2518/Packages/julia-1.2.0/bin/julia ~/research/Force-Constants/permute.jl
