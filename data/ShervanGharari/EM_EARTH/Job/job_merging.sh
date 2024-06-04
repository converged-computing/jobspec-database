#!/bin/bash
#SBATCH --account=rrg-mclark
#SBATCH --ntasks=12 # equal to the number of month
#SBATCH --mem-per-cpu=64G
#SBATCH --time=24:00:00    # time (DD-HH:MM)
#SBATCH --job-name=EM-EARTH-MERG
#SBATCH --error=errors1

# load needed modules
module load StdEnv/2020 gcc/9.3.0 openmpi/4.0.3
module load gdal/3.5.1 libspatialindex/1.8.5
module load python/3.8.10 scipy-stack/2022a mpi4py/3.0.3

# create virtual env inside the job
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index requests
pip install --no-index base64
pip install --no-index hashlib
pip install --no-index glob
pip install --no-index xarray
pip install --no-index matplotlib
pip install --no-index jupyter
pip install --no-index re
pip install --no-index netcdf4
pip install --no-index h5netcdf
pip install --no-index gdown
pip install --no-index dask


python ../code/merging_EM_EARTH.py
