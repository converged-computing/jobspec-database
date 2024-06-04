#!/bin/bash

#PBS -l nodes=1:ppn=1,walltime=4:00:00
#PBS -N DataMaterialize
#PBS -M htw230@nyu.edu
#PBS -m abe
#PBS -e localhost:/home/htw230/${PBS_JOBNAME}.e${PBS_JOBID}
#PBS -o localhost:/home/htw230/${PBS_JOBNAME}.o${PBS_JOBID}

cd /home/htw230/MusicClassification

module load scipy/intel/0.13.3
python pythonrize.py

exit 0;
