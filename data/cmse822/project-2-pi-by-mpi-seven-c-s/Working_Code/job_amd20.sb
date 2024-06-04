#!/bin/bash
########## Define Resources Needed with SBATCH Lines ##########
#SBATCH --array=0-2
#SBATCH --time=00:15:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=64                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=1          # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem=2G                    # memory required per node - amount of memory (in bytes)
#SBATCH --job-name calc_pi     # you can give your job a name for easier identification (same as -J)
#SBATCH --output=output/pi_job_%A_%a.out
#SBATCH --error=output/pi_job_%A_%a.err
 

########## Command Lines to Run ##########

# module purge
# module load GCC/7.3.0-2.30 OpenMPI Python git
  
cd ~/Documents/project-2-pi-by-mpi-seven-c-s/Nick_Work            ### change to the directory where your code is located

num_darts=(1000 1000000 1000000000)
for cpus in 1 2 4 8 16 32 64
do 
    output=$(mpiexec -n $cpus ./pi ${num_darts[$SLURM_ARRAY_TASK_ID]})
    echo -e "${output},${cpus},${num_darts[$SLURM_ARRAY_TASK_ID]} "
done
