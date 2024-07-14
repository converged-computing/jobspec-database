# stuff to setup env
export HDF5_USE_FILE_LOCKING=FALSE # turn off file locking for HDF5 error. Reduces errors.
export AUSRAIN=~st7295/aus_rain_analysis
module load intel-compiler/2021.10.0
module load netcdf/4.7.3 # so have ncdump
module load cdo # cdo!
module load ncview
module load python3/3.11.7
module load openmpi/4.1.5
module load hdf5/1.12.2
module load R/4.2.2
module load python3-as-python # give me python 3!
module load gdal # needed for gdal stuff.
module load nco # for nco
# and activate the virtual env
source $AUSRAIN/venv/bin/activate
# add in dask magic
module use /g/data/hh5/public/modules/
module load dask-optimiser
# paths and stuff
export PYTHONPATH=$PYTHONPATH:~/common_lib:$AUSRRAIN
export PATH=$AUSRAIN:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/apps/R/4.2.2/lib64/R/lib/ # so rpy2 works
# give some info to user.
echo "setup complete."
module list
echo  "Virtual env is $VIRTUAL_ENV"

