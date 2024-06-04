#!/bin/bash
#SBATCH --job-name=dask_gridding
#SBATCH --time=02:00:00           
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8000M
#SBATCH --mail-type=ALL                      # send all mail (way to much)
#SBATCH --mail-user=andrew.d.nolan@maine.edu # email to spend updates too
#SBATCH --output=dask_%A_%a.out              # standard output
#SBATCH --error=dask_%A_%a.err               # standard error

# numbers of cores each job in the array will have
export NUM_WORKERS=32
# use a single thread per cpu core
export THREADS_PER_WORKER=1

source ../../config/modulefile.cc.cedar

# load the source functions
source ./periodic_surge.sh

KEY='crmpt12'
# set up the dask cluter for the instance of the job array
create_dask_cluster

run_name="crmpt12_dx_50_TT_9000.0_MB_-0.37_OFF_Tma_-8.5_B_1.000e-04_SP_2_QP_28"

# rename the file based on the restart point and integration length
new_name="crmpt12_dx_50_TT_0--9ka_MB_-0.37_OFF_Tma_-8.5_B_1.000e-04_SP_2_QP_28"

# rename restart files
mv "result/${KEY}/mesh_dx50/${run_name}.result"\
    "result/${KEY}/mesh_dx50/${new_name}.result"

# rename the "raw" netcdf files
mv "result/${KEY}/nc/${run_name}.nc"\
    "result/${KEY}/nc/${new_name}.nc"

# overwrite the runname with the updated name,
# we postprocess after renaming so the files packed into the zarr archieves have 
# the correct filenames when unpacked
run_name="${new_name}"

# run the post processing commands
post_proccess