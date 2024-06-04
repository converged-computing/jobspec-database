#!/bin/sh


# Alexander Vargo
# 12 June 2017
#
# Run for a mesh of parameters
# for a fixed value of lamb
#
# Updated 15 March 2018 for array jobs
# Updated 16 April 2019 for using the correct version of python

#### PBS preamble

#PBS -N smallMeshFast-rf

#PBS -M ahsvargo@umich.edu
#PBS -m abe

# #cores, memory, walltime
#PBS -l nodes=1:ppn=2,mem=4gb
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -V

# make it an array job
#PBS -t 1-10

# allocation information
#PBS -A annacg_fluxod
#PBS -q fluxod
#PBS -l qos=flux

#### END PBS preamble


# Show list of CPUs this is running on:
cat "CPUs used:"
if [ -s "$PBS_NODEFILE" ]; then 
    echo "Running on"
    uniq -c $PBS_NODEFILE 
fi

# change to the directory we submitted from:
if [ -d "$PBS_O_WORKDIR" ]; then 
    cd $PBS_O_WORKDIR 
    echo "Running from $PBS_O_WORKDIR"
fi

# Run the job
rm -rf __pycache__/
source activate r35py37
/home/ahsvargo/miniconda3/envs/r35py37/bin/python lambTest-CV-fast.py 

