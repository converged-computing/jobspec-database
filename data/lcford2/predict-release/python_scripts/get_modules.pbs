#!/bin/bash
#IMPORTANT! If above shell isn't the same as your account default shell, add
# the "-i" option to the above line to be able to use "module" commands. 

### set the number of nodes
### set the number of PEs per node
### set the XE feature
#PBS -l nodes=1:ppn=1:xe
### set the wallclock time
#PBS -l walltime=01:00:00
### set the job name
#PBS -N get_modules_
### set the job stdout and stderr
#PBS -e ./qsub_output/$PBS_JOBID.err
#PBS -o ./qsub_output/$PBS_JOBID.out
### set email notification
##PBS -m bea
##PBS -M username@host
### In case of multiple allocations, select which one to charge
##PBS -A xyz
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027

# NOTE: lines that begin with "#PBS" are not interpreted by the shell but ARE
# used by the batch system, wheras lines that begin with multiple # signs,
# like "##PBS" are considered "commented out" by the batch system
# and have no effect.

# If you launched the job in a directory prepared for the job to run within,
# you'll want to cd to that directory
# [uncomment the following line to enable this]
cd $PBS_O_WORKDIR

# Alternatively, the job script can create its own job-ID-unique directory
# to run within.  In that case you'll need to create and populate that
# directory with executables and perhaps inputs
# [uncomment and customize the following lines to enable this behavior]
# mkdir -p /scratch/sciteam/$USER/$PBS_JOBID
# cd /scratch/sciteam/$USER/$PBS_JOBID
# cp /scratch/job/setup/directory/* .

# To add certain modules that you do not have added via ~/.modules
# use a line like the following (uncommented, of course). 
#module load craype-hugepages2M perftools 

### launch the application
### redirecting stdin and stdout if needed
### set OMP_NUM_THREADS and the depth accordingly
### in the following there will be 1 MPI task per bulldozer FP module,
### with 2 OMP threads 1 per integer core.

### NOTE: (the "in" file must exist for input)

## export OMP_NUM_THREADS=2
# module load PrgEnv-cray
# module load bwpy
# source /u/eot/lcford2/projects/predict-release/bwvenv/bin/activate
# aprun -b -n 1  bwpy-environ python multi_basin_simple.py colorado Hoover
module list
### For more information see the man page for aprun
