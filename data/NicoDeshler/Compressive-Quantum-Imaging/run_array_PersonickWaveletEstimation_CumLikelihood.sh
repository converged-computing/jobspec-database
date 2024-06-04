#!/bin/bash
#PBS -N QMWaveletEst_CumLikelihood8x8sparse
#PBS -j oe
#PBS -d /home/nicolas/Research/Compressive-Quantum-Imaging/
#PBS -l nodes=1:ppn=8
#PBS -l walltime=24:00:00
#PBS -l mem=16GB
#PBS -t 1-100
matlab "PersonickWaveletEstimation_CumLikelihood(getenv('PBS_ARRAYID'))"
