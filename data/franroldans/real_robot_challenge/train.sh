#!/bin/bash -l
#SBATCH --job-name=test
# specify number of nodes 
#SBATCH -N 1

# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 2

# specify the walltime e.g 20 mins
#SBATCH -t 00:05:00

# set to email at start,end and failed jobs
# SBATCH --mail-type=ALL
# SBATCH --mail-user=robert.mccarthy@ucdconnect.ie

# run from current directory
cd $SLURM_SUBMIT_DIR

# export FI_PROVIDER=verbs
# module load intel/intel-cc
# module load intel/intel-mkl
# module load intel/intel-mpi

source ~/.bash_profile
module load singularity/3.5.2

# The following commands are informed by:
# https://people.tuebingen.mpg.de/felixwidmaier/rrc2021/singularity.html#build-and-run-code-in-singularity

singularity shell --cleanenv --no-home -B /home/people/16304643/robochallenge/workspace /home/people/16304643/robochallenge/rrc2021_latest.sif

source /setup.bash
cd /home/people/16304643/robochallenge/workspace
colcon build
source install/local_setup.bash

mpirun -np 2 ros2 run rrc_example_package testenv 2>&1 | tee testenv.log