#!/bin/bash -l
#
# allocate 1 nodes (4 CPUs) for 3 hours
#PBS -l nodes=1:ppn=4:gtx1080ti,walltime=00:10:00
#
#Mail bei abbruch
#PBS -m a
# job name
#PBS -N CNN_Training
# stdout and stderr files
#PBS -o /home/hpc/capm/sn0515/UVWireRecon/logs/${PBS_JOBID}.out
#PBS -e /home/hpc/capm/sn0515/UVWireRecon/logs/${PBS_JOBID}.err
#
# first non-empty non-comment line ends PBS options

# jobs always start in $HOME -
#source $HPC/.bash_profile

CodeFolder=$HPC/UVWireRecon/
cd ${CodeFolder}

# run
echo -e -n "Start:\t" && date

module load python/2.7-anaconda
# load conda virtualenv
source activate tensorflow
# required:
module use -a /home/vault/capn/shared/apps/U16/modules

echo "$CodeFolder/wrapper.sh $config"
#bash wrapper.sh $config

wait

echo -e -n "Ende:\t" && date 

qdel ${PBS_JOBID}
