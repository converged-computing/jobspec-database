#!/bin/bash
#PBS -l nodes=1:ppn=1:dc2,walltime=0:10:00
#PBS -N viewlifeconnectome
#PBS -V

[ $PBS_O_WORKDIR ] && cd $PBS_O_WORKDIR

module load matlab

rm -rf tracts

export MATLABPATH=$MATLABPATH:$SERVICE_DIR
matlab -nodisplay -nosplash -r main

if [ -s tracts/20.json ];
then
	echo 0 > finished
else
	echo "output missing"
	echo 1 > finished
	exit 1
fi
