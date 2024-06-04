#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -t 30:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 22
#SBATCH -A phy210030p
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=akshay2
#SBATCH --output=/ocean/projects/phy210030p/akshay2/Slurm_logs/prepsubband_slurm_%j.log

# Ensure that the output directory to SBATCH exists prior to batch script execution.

# Define environment variables. $PROJECT = /ocean/projects/<group id>/<username>
SINGULARITY_CONT=$PROJECT/psrsearch.sif
CMDDIR=$PROJECT/HPC_pulsar/cmd_files

# Run dedispersion module within singularity container.
singularity exec -B /local $SINGULARITY_CONT $CMDDIR/dedispersion.cmd
