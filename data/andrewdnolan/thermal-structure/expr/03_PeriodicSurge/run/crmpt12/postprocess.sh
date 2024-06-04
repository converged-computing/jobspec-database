#!/bin/bash
#SBATCH --job-name=dask_gridding
#SBATCH --time=06:00:00           
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8000M
#SBATCH --mail-type=END                      # send all mail (way to much)
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

parse_params()
{   
  IFS=" " read -r QP beta NT T_0<<< $(echo $line | cut -d " " -f 7,9,11,13)
}

KEY='crmpt12'

# set up the dask cluter for the instance of the job array
create_dask_cluster

# # post process the surge files
# for file in $(find ./result/${KEY}/nc/ -name "${KEY}_dx_50_TT_3000.0*B_*.nc"); do 
#     # get the base filename ,with no path info 
#     fn="${file##*/}"
#     # strip the file extension, to get the runname 
#     run_name="${fn%%.nc}"
#     # run the post processing commands
#     post_proccess
# done 


while IFS="" read -r line || [ -n "$line" ]; do
    # only parse the parameters that actually vary 
    parse_params $line
    # get the run name based on parsed parameters for the i-th run
    # NOTE: default value which aren't varied are hard coded
    run_name="${KEY}_dx_50_TT_${NT}.0_MB_-0.36_OFF_Tma_-8.5_B_${beta}_SP_2_QP_${QP}"

    # based on the start time (T_0) and time intergration length (NT)
    # caluculate the final time in kiloyears
    T_f=$(awk -v T_0=$T_0 -v NT=$NT 'BEGIN {print (T_0 + NT)/1e3}')
    # convert the start time from years to kiloyears
    T_0=$(awk -v T_0=$T_0 'BEGIN {print T_0/1e3}')

    # rename the file based on the restart point and integration length
    new_name="${KEY}_dx_50_TT_${T_0}--${T_f}ka_MB_-0.36_OFF_Tma_-8.5_B_${beta}_SP_2_QP_${QP}"

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


done <run/${KEY}/${KEY}.commands
