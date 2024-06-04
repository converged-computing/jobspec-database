#!/bin/bash
#PBS -l nodes=1:ppn=1,mem=250mb,walltime=12:00:00
#PBS -V
#PBS -t 11-20

cd $PBS_O_WORKDIR

#set the dataset here
Dataset={\'13_9\',\'14_7\'}
# set the logname here
logname=output$PBS_ARRAYID.log 

# check everythings okay with some echoes
echo $Dataset
echo $PBS_ARRAYID
echo $logname

module load matlab
matlab -nojvm -logfile $logname -r "SaveNumber = $PBS_ARRAYID; Dataset = $Dataset; ParameterFittingScript;"
