#!/bin/sh -l
#SBATCH -A hpc-wof1
#SBATCH --partition=xjet,ujet,vjet,tjet,kjet
#SBATCH -J build_tables
#SBATCH --ntasks=1 --cpus-per-task=12
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH --output=buildtable_%j.log

#set -eux

time1=$(date '+%s')
echo "Job Started: $(date). Job Id:  $SLURM_JOBID"
echo " "

module use ROOTDIR/modules
module load build_jet_intelTed


srun -n 1 /lfs4/NAGAPE/hpc-wof1/ywang/MPAS/MPAS-Model/build_tables

time2=$(date '+%s')

let diff=time2-time1
let hour=diff/3600
let diff=diff%3600
let min=diff/60
let sec=diff%60

echo -n "Job   Ended: $(date). "
printf 'Job run time:  %02d:%02d:%02d' $hour $min $sec
echo " "
