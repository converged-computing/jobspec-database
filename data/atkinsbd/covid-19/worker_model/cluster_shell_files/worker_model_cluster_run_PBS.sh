#!/bin/bash

#Name the job
#PBS -N worker_model

# Options for output and error files
#PBS -j oe

# Use the submission environment
#PBS -V

#Specify walltime, cores, memory
#PBS -l walltime=10:00:00
#PBS -l ncpus=1
#PBS -l mem=2GB

#Set up job array
#PBS -t 1-2

# Point to the path where this script was submitted from
cd  $PBS_O_WORKDIR

echo PBS: node file is $PBS_NODEFILE

# args/ARGS list
# args[1] JOB_ID
# args[2] RNGseed: To be used to initialise the random number generator
# args[3] COUNTFINAL: Number of replicates requested
# args[4] ENDTIME: Timesteps for each individual simulation
# args[5] n_nodes: Overall size of network
# args[6] n_sectors: Defining total number of work sectors
# args[7] runsets: Scenarios to run: [configuration, intervention] (see load_configurations() and load_interventions() for options)
julia1.4 worker_model.jl ${PBS_ARRAYID} 1 100 10 365 10000 41 '["default" "none"]'

# Resave variables to reduce the filesize
matlab -nodisplay -nosplash -nodesktop -r "resave_MAT_file('../../Results/worker_model/',${PBS_ARRAYID});exit"


echo "Finished"
