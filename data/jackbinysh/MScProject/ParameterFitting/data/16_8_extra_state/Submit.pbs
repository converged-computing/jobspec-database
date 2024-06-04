#!/bin/bash
#PBS -l nodes=1:ppn=4,mem=2000mb,walltime=8:00:00
#PBS -V
#PBS -t 1-50

cd $PBS_O_WORKDIR

echo "$PBS_ARRAYID"

module load matlab
matlab -nojvm -logfile output$PBS_ARRAYID.log -r "SaveNumber = $PBS_ARRAYID; ParameterFittingScript"
