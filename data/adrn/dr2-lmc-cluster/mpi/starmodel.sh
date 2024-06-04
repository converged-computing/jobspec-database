#!/bin/bash
#SBATCH -J isochrones                 # job name
#SBATCH -o isochrones.o%j             # output file name (%j expands to jobID)
#SBATCH -e isochrones.e%j             # error file name (%j expands to jobID)
#SBATCH --array=0-418
#SBATCH -n 1                          # number of cores (not nodes!)
#SBATCH -p cca                        # add to the CCA queue
#SBATCH -t 04:00:00                   # run time (hh:mm:ss)

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/home/apricewhelan/software/lib/
cd /mnt/ceph/users/apricewhelan/projects/dr2-lmc-cluster/scripts

module load gcc openmpi2

date

python run_isochrones_sample.py --index=$SLURM_ARRAY_TASK_ID

date
