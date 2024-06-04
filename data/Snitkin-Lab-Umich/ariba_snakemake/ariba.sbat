#!/bin/sh
# Job name
#SBATCH --job-name=ariba
# User info
#SBATCH --mail-user=UNIQNAME@umich.edu
#SBATCH --mail-type=BEGIN,END,NONE,FAIL,REQUEUE
#SBATCH --export=ALL
#SBATCH --partition=standard
#SBATCH --account=esnitkin1
# Number of cores, amount of memory, and walltime
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1g --time=10:00:00
#  Change to the directory you submitted from
cd $SLURM_SUBMIT_DIR
echo $SLURM_SUBMIT_DIR

# Load modules

# Job commands
snakemake --latency-wait 90 --profile config -s snakefile
