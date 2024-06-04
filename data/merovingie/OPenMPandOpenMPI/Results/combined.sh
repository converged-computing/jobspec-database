#!/bin/bash
#SBATCH --open-mode=append
#SBATCH --nodes=8  ## number of nodes to use to restrict tasks to
#SBATCH --ntasks=8 ## number of tasks (analyses) to run
#SBATCH --output slurm-8-12-1000.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --cpus-per-task=1  ## the number of threads allocated to each task
#SBATCH --mem-per-cpu=500M   # memory per CPU core
#SBATCH --time=00-00:06:00  ## time for analysis (day-hour:min:sec)
#SBATCH --mail-user mikhaelr@chop.edu  ## your email address
#SBATCH --mail-type BEGIN  ## slurm will email you when your job starts
#SBATCH --mail-type END  ## slurm will email you when your job ends
#SBATCH --mail-type FAIL  ## slurm will email you when your job fails

unset LD_PRELOAD

## Load modules
module load openmpi4/gcc/4.0.5
## Insert code, and run your programs here (use 'srun').
srun --mpi=pmix_v3 -n 8 ./combined 8 12 1000
# End of submit file
