#!/bin/bash
#SBATCH --job-name=small_batch        # Job name
#SBATCH --mail-type=END,FAIL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=matt.dale@york.ac.uk     # Where to send mail  
#SBATCH --ntasks=1                          # Run a single task   
#SBATCH --mem=10gb                        # Job memory request
#SBATCH --time=04:00:00                  # Time limit hrs:min:sec
#SBATCH --output=ch_%A_%a.log        # Standard output and error log
#SBATCH --account=cs-charcsi-2019        # Project account
#SBATCH --array=1-20

echo My working directory is `pwd`
echo Running job on host:
echo -e '\t'`hostname` at `date`
echo $SLURM_CPUS_ON_NODE CPU cores available

echo Running array job index $SLURM_ARRAY_TASK_ID, on host:
echo

module load math/MATLAB/2018a

matlab -r 'viking_charc($SLURM_ARRAY_TASK_ID,0,1,4)'
matlab -r 'viking_charc($SLURM_ARRAY_TASK_ID,1,1,4)'
matlab -r 'viking_charc($SLURM_ARRAY_TASK_ID,0,1,6)'
matlab -r 'viking_charc($SLURM_ARRAY_TASK_ID,1,1,6)'
matlab -r 'viking_charc($SLURM_ARRAY_TASK_ID,0,1,8)'
matlab -r 'viking_charc($SLURM_ARRAY_TASK_ID,1,1,8)'
matlab -r 'viking_charc($SLURM_ARRAY_TASK_ID,0,1,10)'
matlab -r 'viking_charc($SLURM_ARRAY_TASK_ID,1,1,10)'

echo
echo Job completed at `date`
