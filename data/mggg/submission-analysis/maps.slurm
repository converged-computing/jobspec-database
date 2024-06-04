#!/bin/bash
#SBATCH --job-name="State_Report_Maps" # Job name
#SBATCH --time=5-00:00:00 # days-hh:mm:ss
#SBATCH --nodes=1 # how many computers do we need?
#SBATCH --ntasks-per-node=1 # how many cores per node do we need?
#SBATCH --mem=5000 # how many MB of memory do we need (5GB here)
#SBATCH --output="/cluster/home/jdesch01/coi-maps/logs/log_%A.txt" # where to save the output file.
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE # when to email you
#SBATCH --mail-user=john.deschler@tufts.edu # the email to use
​
source ~/.bashrc  # need to set up the normal environment.
echo running on: `hostname` # print some info about where we are running
# cd into the correct directory
cd $HOME
cd submission-analysis
conda activate coi-maps
​
python maps_and_lookups.py #$SLURM_ARRAY_TASK_ID # run the python code with the arguments. 