#!/bin/bash

###PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1
###PBS -lselect=1:ncpus=8:mem=48gb:ngpus=2
#PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1
#PBS -lwalltime=24:00:00

#######
## For getting a script without gpu use:
## -> #PBS -lselect=1:ncpus=1:mem=1gb
## For getting a script with a gpu use: 
## -> #PBS -lselect=1:ncpus=16:mem=96gb:ngpus=4:gpu_type=RTX6000
## cpus must be 4*n where n the number of requested gpus and memory is 24*n

#----------------------#
#                      #
#  Setting parameters  #
#                      #
#----------------------#

## Path to python inter
PYTHONCMD="/rds/general/user/sg21/home/miniconda3/bin/python"

## Path to folder containign the code
PROJECTPATH="/rds/general/project/arise/live/PDE_VAE_pytorch_forked"

## Name of the executable file
PYTHONFILE='modified_run.py'

## arguments
ARGS="input_files/KS_train.json"


########
# Code #
########

cd ${PROJECTPATH}
${PYTHONCMD} ${PYTHONFILE} ${ARGS}
