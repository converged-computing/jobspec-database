#!/bin/bash
#PBS -l nodes=3:ppn=16:dc2,walltime=1:00:00
#PBS -N app-dipy-tracking
#PBS -V


#for local testing
if [ -z $SERVICE_DIR ]; then export SERVICE_DIR=`pwd`; fi
#ENV="IUHPC"

[ $PBS_O_WORKDIR ] && cd $PBS_O_WORKDIR

if [ $ENV == "IUHPC" ]; then
	if [ $HPC == "KARST" ]; then
		module unload python
		module load anaconda2
	fi
	if [ $HPC == "CARBONATE" ]; then
		module unload python
		module load anaconda/python2.7
	fi
	
	export PYTHONPATH=$PYTHONPATH:/N/u/aryaam/Karst/github_repos/dipy
	export PYTHONPATH=$PYTHONPATH:/N/u/aryaam/Karst/github_repos/nibabel
	export PYTHONPATH=$PYTHONPATH:/N/u/aryaam/Karst/github_repos/pyAFQ
fi

if [ $ENV == "VM" ]; then
	export PYTHONPATH=$PYTHONPATH:/usr/local/dipy
	export PYTHONPATH=$PYTHONPATH:/usr/local/nibabel
	export PYTHONPATH=$PYTHONPATH:/usr/local/pyAFQ
fi

echo "running main"

#matlab -nodisplay -nosplash -r main
time python $SERVICE_DIR/main.py

ret=$?
if [ $ret -ne 0 ]; then
    echo "main.py failed"
    echo $ret > finished
    exit $ret
fi

if [ -s mapping.nii.gz ];
then 
	echo 0 > finished
else 
	echo "files missing"
	echo 1 > finished
	exit 1
fi
