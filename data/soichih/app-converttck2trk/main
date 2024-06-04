#!/bin/bash
#PBS -l nodes=1:ppn=8:dc2,walltime=0:10:00
#PBS -N tck2trk
#PBS -V

module load singularity 2> /dev/null

singularity exec -e docker://brainlife/dipy:0.13 ./main.py
#singularity exec -e docker://nipype/nipype ./main.py

if [ -s track.trk ] 
then
	echo 0 > finished
else
	echo "failed"
	echo 1 > finished
	exit 1
fi
