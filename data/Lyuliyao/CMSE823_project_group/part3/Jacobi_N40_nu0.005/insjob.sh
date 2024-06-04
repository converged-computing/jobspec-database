#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########
 
#SBATCH --time=24:00:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                   # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=1                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=1           # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem-per-cpu=8G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name Incomp_NS        # you can give your job a name for easier identification (same as -J)
#SBATCH -C amr 

########## Command Lines for Job Running ##########
 
module load intel  ### load necessary modules.
module load FFmpeg
ifort -o kalle_anka.x ins.f90 -mkl

srun ./kalle_anka.x                 ### call your executable. (use srun instead of mpirun.)
matlab -nodisplay -r "pl"
srun ffmpeg -r 10 -i frame"%05d".jpg -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" test.mp4 
scontrol show job $SLURM_JOB_ID     ### write job information to SLURM output file.
js -j $SLURM_JOB_ID                 ### write resource usage to SLURM output file (powertools command).
