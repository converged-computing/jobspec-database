#!/bin/bash
#PBS -l nodes=1:ppn=1,walltime=0:20:00
#PBS -N app-autoalignacpc
#PBS -V

module load singularity
singularity exec -e docker://brainlife/mcr:neurodebian1604-r2017a ./compiled/autoalignmatlab

#export MATLABPATH=$MATLABPATH:$SERVICE_DIR
#matlab -nodisplay -nosplash -r autoalign-matlab

#check for output files
if [ -s t1.nii.gz ];
then
	echo 0 > finished
else
	echo "output t1.nii.gz missing"
	echo 1 > finished
	exit 1
fi

echo $? > finished
