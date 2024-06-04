#!/bin/bash
#SBATCH -J rays
#SBATCH -o /discover/nobackup/wgblumbe/infraGA/rays.out
#SBATCH -e /discover/nobackup/wgblumbe/infraGA/rays.err
#SBATCH --account=s2094
#SBATCH --ntasks=28
#SBATCH --constraint=hasw
#SBATCH --qos=allnccs
#SBATCH --time=00:10:00
## source common file for modules and paths 
source ./common.reg || exit 1
cd /discover/nobackup/wgblumbe/infraGA
#export MPIRUN='mpirun -np 28'
which mpirun
# Set up symbolic links to binaries
mpirun -np 28 ./bin/infraga-accel-3d -prop examples/ToyAtmo.met incl_step=1.0 bounces=2 azimuth=-45.0 write_rays=true min_x=0 max_x=60 min_y=0 max_y=60 write_atmo=true 

