#!/bin/bash
#SBATCH --job-name=nullsim       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --array=1-100%10         # job_array%maximum_to_run_simultaneously
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=00:40:00          # total run time limit (HH:MM:SS)
#SBATCH --output=/mmfs1/scratch/kurkela/output/null_perms_%a.out
#SBATCH --mail-type=end
#SBATCH --mail-user=kurkela@bc.edu

cd /mmfs1/data/kurkela/Desktop/CamCan/code
module load matlab
matlab -nodisplay -nosplash -r "D_cbpm_nullsim('memoryability', 'all', 'default', 0.01, 'none', $SLURM_ARRAY_TASK_ID);"
