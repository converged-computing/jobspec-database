#!/bin/bash
#PBS -q hera
#PBS -j oe
#PBS -N imaging
#PBS -l nodes=1:ppn=1
#PBS -l walltime=4:00:00
#PBS -l mem=6GB
#PBS -M h.garsden@qmul.ac.uk

cd /lustre/aoc/projects/hera/hgarsden/fringe_rate_filter
module load mpi/openmpi-x86_64

~/anaconda3_all/bin/python mk_uvfits.py
