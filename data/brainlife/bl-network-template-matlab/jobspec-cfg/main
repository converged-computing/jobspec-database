#!/bin/bash

#SBATCH --job-name=template
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00

# Copyright (c) 2020 brainlife.io at University of Texas at Austin and Indiana University
# 
# This is the main file run by the brainlife.io orchestration system
#
# Author: Franco Pestilli

# run the actual python code
singularity exec -e docker://brainlife/mcr:r2019a ./compiled/main config.json
