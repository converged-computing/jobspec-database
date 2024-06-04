#!/bin/bash
#SBATCH --job-name=dask_gridding
#SBATCH --time=02:00:00           
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000M
#SBATCH --mail-type=END                      # send all mail (way to much)
#SBATCH --mail-user=andrew.d.nolan@maine.edu # email to spend updates too
#SBATCH --output=dask_%A_%a.out              # standard output
#SBATCH --error=dask_%A_%a.err               # standard error

# numbers of cores each job in the array will have
export NUM_WORKERS=16
# use a single thread per cpu core
export THREADS_PER_WORKER=1

source ../../config/modulefile.cc.cedar

# load the source functions
source ./sensitivity.sh

KEY='crmpt12'
dx=50
NT=30000
dt=0.1

# make the thinned dir for each task in the job array
mkdir "${SLURM_TMPDIR}/thinned"

# check to make sure a thinned dir exists to dump the tar files
if [ ! -d "result/${KEY}/thinned/" ]; then
  mkdir  "result/${KEY}/thinned/"
fi

# set up the dask cluter for the instance of the job array
create_dask_cluster

for file in $(find ./result/crmpt12/nc/ -name "*.nc" -ctime -5); do 

    # get the base filename ,with no path info 
    fn="${file##*/}"
    # strip the file extension, to get the runname 
    run_name="${fn%%.nc}"
    # run the post processing commands
    post_proccess

done 
