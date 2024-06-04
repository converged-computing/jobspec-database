#!/bin/bash
#PBS -l nodes=3:ppn=16:dc2,walltime=0:40:00
#PBS -N app-tract-profile
#PBS -V


#for local testing
if [ -z $SERVICE_DIR ]; then export SERVICE_DIR=`pwd`; fi
#ENV="IUHPC"

[ $PBS_O_WORKDIR ] && cd $PBS_O_WORKDIR

if [ $ENV == "IUHPC" ]; then
	if [ $HPC == "KARST" ]; then
		#not yet yested for brlife@karst (probably don't need anaconda2, dipy, nibabel)
		module unload python
		module load anaconda2
		
		export PYTHONPATH=$PYTHONPATH:/N/u/aryaam/Karst/dipy
		export PYTHONPATH=$PYTHONPATH:/N/u/aryaam/Karst/github_repos/nibabel
		export PYTHONPATH=$PYTHONPATH:/N/u/aryaam/Karst/github_repos/pyAFQ
	fi
	if [ $HPC == "CARBONATE" ]; then
		#deps > pip install dipy nibabel boto3 cloudpickle dask toolz 
		export PYTHONPATH=$PYTHONPATH:/N/u/aryaam/Karst/github_repos/pyAFQ
		export PYTHONPATH=$PYTHONPATH:/N/u/aryaam/Karst/github_repos/nibabel
	fi
fi

if [ $ENV == "VM" ]; then
	#not tested yet
	#deps > pip install dipy nibabel boto3 cloudpickle dask toolz 
	export PYTHONPATH=$PYTHONPATH:/usr/local/pyAFQ
fi

echo "running main"

#matlab -nodisplay -nosplash -r main
time python -u $SERVICE_DIR/main.py
ret=$?
if [ $ret -ne 0 ]; then
    echo "main.py failed"
    echo $ret > finished
    exit $ret
fi

count=$(ls profile/*.png | wc -l)
if [ $count -eq 4 ];
then 
	echo 0 > finished
else 
	echo "files missing"
	echo 1 > finished
	exit 1
fi
