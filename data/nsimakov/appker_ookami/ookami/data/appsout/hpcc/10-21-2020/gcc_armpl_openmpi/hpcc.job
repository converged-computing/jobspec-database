#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -t 24:00:00
#SBATCH -p long
#SBATCH -q long
##SBATCH -w fj[047-174]

pwd


#module restore PrgEnv-cray
#module load cray-mvapich2_nogpu_svealpha
#module load cray-fftw 
module load slurm

#spack load hpcc@develop %cce ^mvapich2 ^cray-libsci
#spack load hpcc@develop fft=fftw3 %cce ^mvapich2 ^cray-libsci

spack load hpcc@develop fft=fftw3 %gcc ^armpl ^openmpi


EXE=${EXE:-$(which hpcc)}
echo $EXE
mpirun -n 48 $EXE 

