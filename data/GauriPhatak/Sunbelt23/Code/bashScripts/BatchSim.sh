#!/bin/bash
#SBATCH -t 2-12:30:00
#SBATCH --job-name=NWSimImp
#SBATCH -N 2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --array=1-16
#SBATCH --constraint=el7

# Specify the path to the config file
config=/nfs/stak/users/phatakg/ResearchCode/Sunbelt23/Code/bashScripts/config.txt

# Extract the sample name for the current $SLURM_ARRAY_TASK_ID
sim=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)

# Extract the sex for the current $SLURM_ARRAY_TASK_ID
type=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)

# load any software environment module required for app (e.g. matlab, gcc, cuda)
module load gcc/12.2
module load R/4.2.2

# run my job (e.g. matlab, python)
Rscript ../NWSimulationGen.R ${sim} ${type}