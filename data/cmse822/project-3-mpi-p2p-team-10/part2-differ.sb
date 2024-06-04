#!/bin/bash
########## Define Resources Needed with SBATCH Lines ##########
 
#SBATCH --time=00:01:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --ntasks-per-node=1                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --nodes=2          # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem=2G                    # memory required per node - amount of memory (in bytes)
#SBATCH --constraint=amd20
#SBATCH --job-name HW3-NON-BLOCKING-DIFFER      # you can give your job a name for easier identification (same as -J)
#SBATCH --exclusive
 
########## Command Lines to Run ##########

module purge
module load intel/2021a
  
cd /mnt/home/kamalida/cmse822/project-3-mpi-p2p-team-10                   ### change to the directory where your code is located
 
mpicxx ./ping-ping-non-blocking.c -o ping-ping-non-blocking.out
mpiexec -n 2 ./ping-ping-non-blocking.out

 
scontrol show job $SLURM_JOB_ID     ### write job information to output file
