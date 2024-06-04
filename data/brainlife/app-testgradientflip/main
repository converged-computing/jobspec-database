#!/bin/bash
#PBS -l nodes=1:ppn=16,vmem=32g,walltime=6:00:00
#PBS -N testgradientflip

module load matlab
module load spm
matlab -nodisplay -nosplash -r main
if [ ! -s results.txt ]; then
	echo "output missing"
	exit 1
fi
