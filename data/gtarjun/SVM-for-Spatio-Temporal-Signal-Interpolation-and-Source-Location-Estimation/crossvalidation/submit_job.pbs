#!/bin/bash


#PBS -l nodes=12:ppn=8
#PBS -l walltime=48:00:00
#PBS -N conn
##PBS -m abe
#PBS -M arjun@unm.edu

JOBS_PER_NODE="1"



## load matlab and GNU parallel
module load matlab/R2017a
module load parallel-20170322-intel-18.0.2-4pa2ap6

echo "Starting a new run on $(date)" > status.log


## change to directory job is submitted from 
cd $PBS_O_WORKDIR

## call GNU parallel to submit conn job for all subjects
#parallel --dryrun --joblog job.log --env PATH -j $PBS_NP --sshloginfile $PBS_NODEFILE -I ,, -a subject_list 'matlab -nodesktop -nodisplay -r "conn_input=,,;conn_test"' > conn.log
parallel   --joblog job.log --workdir $PBS_O_WORKDIR -j $JOBS_PER_NODE --env PATH  --sshloginfile $PBS_NODEFILE -I ,, -a parameters './wrapper1.sh ,, > conn.log'

#matlab -r -nodesktop -nodisplay << EOF
#conn_input='sub-p008';
#conn_test
#EOF
