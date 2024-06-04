#!/bin/bash
#PBS -N ASC_daily
#PBS -P x77
#PBS -q normalbw
#PBS -l walltime=3:00:00
#PBS -l mem=250GB
#PBS -l software=netcdf
#PBS -l ncpus=28
#PBS -l storage=gdata/v45+gdata/hh5+gdata/cj50+gdata/ik11+scratch/x77+scratch/v45+gdata/x77
#PBS -v month,year

#Calculates the monthly Along and cross isobath (1000m) speed monthly, for the specified year
## Note, run this with:
## qsub -v month=3,year=1958 Submit_daily_speed_XYgrid.sh

script_dir=/g/data/x77/wf4500/ASC_project/GH/ASC_SpatialTemporalVariability/Python/queue_logs/
cd $script_dir

# load conda
module use /g/data/hh5/public/modules
module unload conda
module load conda/analysis3-unstable
#For some reason the module here cant find xarray.ufuncs. maybe you just have to find the right module

module list

# run
python3 ../Daily_along_slope_velocities_XYgrid-monthly_output.py $month $year &>> ASCdailyxygrid_${year}_${month}.out


