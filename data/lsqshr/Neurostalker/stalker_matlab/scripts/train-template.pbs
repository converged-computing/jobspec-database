#!/bin/bash
# pbs launching script example for NAMD job
# PLEASE DO NOT CHANGE THIS FILE. TO SUBMIT QUEUE, PLS MAKE A COPY OF THIS FILE AND MAKE THE ACCORDING CHANGES

#     job name:
#PBS -N NEUROSTALKER 
#PBS -P RDS-FEI-NRMMCI-RW
#PBS -q gpu 

#     how many cpus?
#PBS -l ncpus=20

#PBS -l pmem=4000mb

# How long to run the job? (hours:minutes:seconds)
#PBS -l walltime=6:0:0

#     Name of output file:
#PBS -o trainlog.txt

#     Environmental varibles to make it work:
 
module load matlab;
cd $PBS_O_WORKDIR;
 
#     Launching the job!
JOBNAME='sharp-sphere-0.1';

DATETIME=$(date +"%F-%T");

#     Transfer the trained model to data folder
cachefolder=/project/RDS-FEI-NRMMCI-RW/NeurostalkerModelCache/$JOBNAME$DATETIME;

if [ ! -d "$cachefolder" ];then
	echo "Creating cache job folder: $cachefolder";
	mkdir $cachefolder;
else
	echo "Cache job folder exists: $cachefolder";
fi

cp loadTrPara.m $cachefolder/loadTrPara.m;
cp loadGtPara.m $cachefolder/loadGtPara.m;
me="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
cp $me $cachefolder/run.pbs;

#     Run Script
matlab -nodesktop -nodisplay -nosplash -r "run('train.m')";

echo "Dumping in the cached model...";

cp trainlog.txt $cachefolder/trainlog.txt;
cp  ../data/results/trained_models/modelcache.mat $cachefolder/modelcache.mat;