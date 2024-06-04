#!/bin/bash --login

########## SBATCH Lines for Resource Request ##########

#SBATCH --time=00:10:00            # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1              # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=32               # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH -C '[intel18]'


########## Command Lines to Run ##########
 
#module load powertools/1.2 

~/julia-1.7.0/bin/julia generate_mesh.jl 


g++ openmp_version.cpp -fopenmp
./a.out 1 
./a.out 2 
./a.out 4 
./a.out 8 
./a.out 16 
./a.out 32 

~/julia-1.7.0/bin/julia result_figure.jl 


 
scontrol show job $SLURM_JOB_ID     ### write job information to SLURM output file.
js -j $SLURM_JOB_ID                 ### write resource usage to SLURM output file (powertools command).
