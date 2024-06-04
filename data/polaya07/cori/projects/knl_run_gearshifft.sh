#!/bin/bash -l
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -J gearshifft
#SBATCH -C knl,quad,cache 

#Modules' versions
BOOST_VER=1.65.1
FFTW_VER=3.3.6.3

#Load specific modules
module swap PrgEnv-intel/6.0.4  PrgEnv-gnu
module load cray-fftw/$FFTW_VER
module load boost/$BOOST_VER

# -N: nodos -t tiempomax -C filter de procesador. -J name job -q categoria para darle prioridad al trabajo.

CURDIR=$HOME/projects/gearshifft

#Data size from extensions 
FEXTENTS1D=$CURDIR/share/gearshifft/extents_1d_publication.conf
FEXPRUEBA=$CURDIR/share/gearshifft/extents_1d_fftw_copy.conf 
FEXTENTS1DFFTW=$CURDIR/share/gearshifft/extents_1d_fftw.conf  # excluded a few very big ones
FEXTENTS2D=$CURDIR/share/gearshifft/extents_2d_publication.conf
FEXTENTS3D=$CURDIR/share/gearshifft/extents_3d_publication.conf
FEXTENTS=$CURDIR/share/gearshifft/extents_all_publication.conf

#Run with specific FFT
today=$(date +%Y_%m_%d-%H:%M:%S) 

#knl
#time $HOME/projects/gearshifft-intel-knl/bin/gearshifft_fftw -f $FEXTENTS1DFFTW -v -o "$HOME/projects/results/$today-knl_nw.csv" --rigor estimate
#time $HOME/projects/gearshifft-intel-knl/bin/gearshifft_fftw -f $FEXPRUEBA -v -o "$HOME/projects/results/$today-knl_nw.csv" --rigor estimate
#knl-wrappers
time $HOME/projects/gearshifft-intel-knl/bin/gearshifft_fftwwrappers -f $FEXTENTS1DFFTW -v -o "$HOME/projects/results/$today-knl_wc.csv" --rigor estimate
#time $HOME/projects/gearshifft-knl/bin/gearshifft_fftw -e 129x128 -v -o "$HOME/projects/results/$today-knl_nw.csv"


