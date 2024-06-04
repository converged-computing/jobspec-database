#!/bin/bash
#PBS -N QMWaveletEst_4x4
#PBS -j oe
#PBS -d /home/nicolas/Research/Compressive-Quantum-Imaging/
#PBS -l nodes=1:ppn=8
#PBS -l walltime=24:00:00
#PBS -l mem=16GB
#PBS -t 1-100
matlab "PersonickWaveletEstimation(getenv('PBS_ARRAYID'))"
