#!/bin/bash
#SBATCH -p RM
#SBATCH -t 32:00:00
#SBATCH -N 3
#SBATCH --ntasks-per-node 128
#SBATCH -A phy210030p
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=akshay2
#SBATCH --output=/ocean/projects/phy210030p/akshay2/Slurm_logs/RM_accelsearch_slurm_%j.log

# Ensure that the output directory to SBATCH exists prior to batch script execution.

# Define environment variables. $PROJECT = /ocean/projects/<group id>/<username>
SINGULARITY_CONT=$PROJECT/psrsearch.sif
CMDDIR=$PROJECT/HPC_pulsar/cmd_files

module swap intel pgi
module load mpi/pgi_openmpi

# Run acceleration searches within singularity container.
mpirun -n $SLURM_NTASKS singularity exec -B /local $SINGULARITY_CONT \
	python /ocean/projects/phy210030p/akshay2/HPC_pulsar/executables/accelsearch_sift_fold.py \
       -i /ocean/projects/phy210030p/akshay2/HPC_pulsar/config/accel.cfg
