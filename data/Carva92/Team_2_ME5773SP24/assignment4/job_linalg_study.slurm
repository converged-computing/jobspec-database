#!/bin/bash
# 
#
# Authors: Julian Carvajal Rico
#          James Platt Standard
#          Roberto Enriquez Vargas
#
# ======================================================================
#SBATCH -J linalg_study
#SBATCH -o outFile.%j.txt    # Name of 'stdout' output file.
#SBATCH -e errFile.%j.txt    # Name of 'stderr' error file.
#SBATCH -p compute1          # Partition
#SBATCH -N 1                 # Total number of nodes to be requested.
#SBATCH -n 1                 # Total number of tasks to be requested.
#SBATCH -c 80                # Number of threads used by each task.
#SBATCH -t 00:05:00          # Maximum estimated run time (dd-hh:mm:ss)
#SBATCH --mail-type=ALL      # Mail events to notify (END, FAIL, ALL).
#SBATCH --mail-user your_@email.edu # Put your utsa-email here.
#
# 
# # MKL is Intel's Math Kernel Library. One of the environment 
# # variables that control the execution is the following:
# export MKL_NUM_THREADS=1
# 

echo "Starting job_linalg_study.slurm"


# Load Anaconda3
module load anaconda3

# Acivate the environment
conda activate envTeam2


MKL_VALUES=(1 2 4 8 16 20 40)

# Loop to go over all MKL_VALUES
for MKL_NUM_THREADS in "${MKL_VALUES[@]}"
do
    echo "Running with MKL_NUM_THREADS=$MKL_NUM_THREADS"
    # Set the MKL_NUM_THREADS
    export MKL_NUM_THREADS=$MKL_NUM_THREADS
    
    # Run the linalg.py script
    srun --exclusive -N1 -n1 -c $SLURM_CPUS_PER_TASK python3 linalg.py &
    
    wait
done