#!/bin/bash


#PBS -l nodes=50:ppn=8
#PBS -l walltime=48:00:00
#PBS -N conn_ns_gp
##PBS -m abe
#PBS -M arjun@unm.edu

JOBS_PER_NODE=8



## load matlab and GNU parallel
module load matlab/R2017a
module load parallel-20170322-intel-18.0.2-4pa2ap6

echo "Starting a new run on $(date)" > status.log


## change to directory job is submitted from 
cd $PBS_O_WORKDIR

## call GNU parallel to submit conn job for all subjects
#parallel --dryrun --joblog job.log --env PATH -j $PBS_NP --sshloginfile $PBS_NODEFILE -I ,, -a subject_list 'matlab -nodesktop -nodisplay -r "conn_ss_gp_input=,,;conn_ss_gp_test"' > conn_ss_gp.log
parallel   --joblog job.log --workdir $PBS_O_WORKDIR  --env PATH  --sshloginfile $PBS_NODEFILE -I ,, -a parameters './wrapper.sh ,, > conn.log'

## matlab -r -nodesktop -nodisplay << EOF
## conn_ss_gp_input='sub-p008';
## conn_ss_gp_test
## EOF
