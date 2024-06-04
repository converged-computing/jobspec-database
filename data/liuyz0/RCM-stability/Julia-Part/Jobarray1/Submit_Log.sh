#!/bin/bash

#SBATCH -o Distributed_SimuLin.log-%j
#SBATCH -n 48 # Number of tasks

# Initialize the module command first source
source /etc/profile

# Load cuda and Julia Module
module load julia/1.8.5

# Call your script as you would from the command line
julia Simulation_distrv1Log.jl