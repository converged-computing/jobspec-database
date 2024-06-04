#!/bin/bash --login

########## SBATCH Lines for Resource Request ##########

#SBATCH --time=4:00:00            # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1              # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=32# number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH -C '[amd20]'
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10GB

########## Command Lines to Run ##########
 
#module load powertools/1.2 
ml AOCC/2.2.0 
ml OpenMPI
srun  -n 1 ~/julia-1.7.0/bin/julia MPI_code.jl 
srun  -n 2 ~/julia-1.7.0/bin/julia MPI_code.jl 
srun  -n 4 ~/julia-1.7.0/bin/julia MPI_code.jl 
srun  -n 8 ~/julia-1.7.0/bin/julia MPI_code.jl 
srun  -n 16 ~/julia-1.7.0/bin/julia MPI_code.jl 
srun  -n 32 ~/julia-1.7.0/bin/julia MPI_code.jl 



 
scontrol show job $SLURM_JOB_ID     ### write job information to SLURM output file.
js -j $SLURM_JOB_ID                 ### write resource usage to SLURM output file (powertools command).
