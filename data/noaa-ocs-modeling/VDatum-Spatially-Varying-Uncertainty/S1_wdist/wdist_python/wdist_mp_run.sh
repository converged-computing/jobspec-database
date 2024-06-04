#!/bin/bash -l
#
# -- Request that this job run on whichJet
#SBATCH --partition=tjet,ujet,sjet,vjet,xjet,kjet
# -- Specifying the project account
#SBATCH -A vdatum
#SBATCH -J hello
#SBATCH -c 1
#
# -- Specify where to put stdout and stderr
#SBATCH -o station-%a.out
#SBATCH -D .
#PBS -M elena.tolkova@noaa.gov
#
# -- Specify a maximum wallclock of 3 hours
#SBATCH --time=03:00:00
#
# -- Set the name of the job, or Slurm will default to the name of the script
#SBATCH --job-name=HPL
#SBATCH -v
#SBATCH --array=0-9
#SBATCH --qos batch 
#SBATCH --mem=20000M
#SBATCH --ntasks=1
date
echo "host name is `hostname` : $SLURM_ARRAY_TASK_ID gages_ids_nodes.txt fort.14"
cp   ../TXstabilize/fort.14 tt/${SLURM_ARRAY_TASK_ID}.fort.14
cp   gages_ids_nodes.txt tt/${SLURM_ARRAY_TASK_ID}.gages.txt
./wdist.py $SLURM_ARRAY_TASK_ID tt/${SLURM_ARRAY_TASK_ID}.gages.txt tt/${SLURM_ARRAY_TASK_ID}.fort.14
date
