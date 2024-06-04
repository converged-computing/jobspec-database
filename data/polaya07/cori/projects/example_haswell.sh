#!/bin/bash -l
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -C haswell
#SBATCH -t 00:10:00
#SBATCH -J gearshifft
#SBATCH --output=./outputs/%j.out
#knl SBATCH -C knl,quad,cache 

#Modules' versions
BOOST_VER=1.67.0
FFTW_VER=3.3.6.5

#Load specific modules
module swap PrgEnv-intel PrgEnv-gnu
module load cray-fftw/$FFTW_VER
module load boost/$BOOST_VER

# -N: nodos -t tiempomax -C filter de procesador. -J name job -q categoria para darle prioridad al trabajo.

CURDIR=$HOME/projects/gearshifft

#Data size from extensions
FEXUNIQUE=$CURDIR/share/gearshifft/extents_uni.conf
FEXSMALL=$CURDIR/share/gearshifft/extents_small.conf
FEXLARGE=$CURDIR/share/gearshifft/extents_1d_fftw_large.conf
FEXCOMPLETE=$CURDIR/share/gearshifft/extents_1d_fftw_copy.conf
FEXPRUEBA=$CURDIR/share/gearshifft/extents_1d_fftw.conf
FEXTENTS1D=$CURDIR/share/gearshifft/extents_1d_publication.conf
FEXTENTS1DFFTW=$CURDIR/share/gearshifft/extents_1d_fftw.conf  # excluded a few very big ones
FEXTENTS2D=$CURDIR/share/gearshifft/extents_2d_publication.conf
FEXTENTS3D=$CURDIR/share/gearshifft/extents_3d_publication.conf
FEXTENTS=$CURDIR/share/gearshifft/extents_all_publication.conf

#Run with specific FFT
today=$(date +%Y_%m_%d-%H:%M:%S)

#gnu-nwgg
#time $HOME/projects/gearshifft-gnu-nw/bin/gearshifft_fftw -f $FEXTENTS1DFFTW -v -o "$HOME/projects/results/$today-gnu_nw-1d.csv" --rigor estimate

#Running multiple signal sizes
#time $HOME/projects/gearshifft-gnu-wc/bin/gearshifft_fftwwrappers -f $FEXUNIQUE -v -o "$HOME/projects/Results/$today-gnu_wc.csv" --rigor estimate

#Measure power and time

#Data File = 8192 (65K)
size=8192
echo $SLURM_JOBID	
mkdir $SLURM_JOBID
cd $SLURM_JOBID 
mkdir $size
cd $size
echo $(date +%Y_%m_%d-%H:%M:%S)
$HOME/projects/powercap/powercap_plot &
sleep 0.5
echo $(date +%Y_%m_%d-%H:%M:%S)
time $HOME/projects/gearshifft-gnu-wc/bin/gearshifft_fftw -e $size -r */double/*/*_Real -v -o "$HOME/projects/$SLURM_JOBID/$size/haswell_wc.csv" --rigor estimate
sleep 0.5
pkill -KILL -u polaya $HOME/projects/powercap/powercap_plot
echo $(date +%Y_%m_%d-%H:%M:%S)
kill %1


