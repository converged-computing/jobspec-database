#!/bin/bash
for (( i = 1; i <= $1; i++ ))      ### Outer for loop ###
do
	
	sleep 3;    	
	for (( j = 1 ; j <= $2; j++ )) ### Inner for loop ###
    do
	    sleep 3;
sbatch <<EOT
#!/bin/bash
#SBATCH --open-mode=append
#SBATCH --nodes=$i  ## number of nodes to use to restrict tasks to
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH --output slurm-1.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --cpus-per-task=1  ## the number of threads allocated to each task
#SBATCH --mem-per-cpu=500M   # memory per CPU core
#SBATCH --time=00-00:06:00  ## time for analysis (day-hour:min:sec)
#SBATCH --hint=nomultithread  ## disable in-core multi-threading (Hyper-threading/SMP)
#SBATCH --mail-user mikhaelr@chop.edu  ## your email address
#SBATCH --mail-type BEGIN  ## slurm will email you when your job starts
#SBATCH --mail-type END  ## slurm will email you when your job ends
#SBATCH --mail-type FAIL  ## slurm will email you when your job fails

unset LD_PRELOAD
#---#
echo "The job "${SLURM_JOB_ID}" is running on "${SLURM_JOB_NODELIST}
#---#



## Load modules
module load openmpi4/gcc/4.0.5
## Insert code, and run your programs here (use 'srun').
	echo $i $j
	srun --exclusive --mpi=pmix_v3 -n $i ./combined $i $j 1

# End of submit file
EOT
    done
  echo "" #### print the new line ###
done

