#!/bin/bash

#PBS -N uppercase
#PBS -l select=1:ncpus=2:ngpus=1:mem=16gb:scratch_local=5gb:cluster=adan -q gpu
#PBS -l walltime=10:00:00
#PBS -M martin.studna2@gmail.com
#PBS -m abe
# The 4 lines above are options for scheduling system: job will run 1 hour at maximum, 1 machine with 4 processors + 4gb RAM memory + 10gb scratch memory are requested, email notification will be sent when the job aborts (a) or ends (e)

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR=/storage/brno2/home/martin-studna/testdirectory # substitute username and path to to your real username and path

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually 
#echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt


export NEPTUNE_API_TOKEN=""

#loads the Gaussian's application modules, version 03
module add python/3.8.0-gcc-rab6t
module add cudnn-7.6.4-cuda10.0 cudnn-7.6.4-cuda10.1
module add tensorflow-2.0.0-gpu-python3
pip3 install neptune-client

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# copy input file "h2o.com" to scratch directory
# if the copy operation fails, issue error message and exit
cp $DATADIR/*  $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }



# move into scratch directory
cd $SCRATCHDIR 

# run Gaussian 03 with h2o.com as input and save the results into h2o.out file
# if the calculation ends with an error, issue error message an exit
python3 uppercase.py || { echo >&2 "Calculation ended up erroneously (with a code $?) !!"; exit 3; }

# move the output to user's DATADIR or exit in case of failure
cp uppercase_test.txt $DATADIR/ || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

# clean the SCRATCH directory
clean_scratch
