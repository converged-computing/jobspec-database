#!/bin/bash
#PBS -N batch_job
#PBS -l select=1:mem=32gb:scratch_local=20gb:ngpus=1:gpu_cap=cuda60
#PBS -l walltime=20:00:00
#PBS -q gpu

DATADIR=/storage/plzen1/home/gajdoma6

test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

cp -r $DATADIR/cluster/run_several_architectures.py $DATADIR/cluster/architectures_list.py $DATADIR/cluster/classes.py $DATADIR/cluster/postprocess.py $SCRATCHDIR
cd $SCRATCHDIR

mkdir ./raw_models
mkdir ./histories

singularity exec -B $SCRATCHDIR:/scratchdir -B $DATADIR/data:/data --nv /cvmfs/singularity.metacentrum.cz/NGC/TensorFlow\:23.05-tf2-py3.SIF python /scratchdir/run_several_architectures.py

mv ./raw_models $DATADIR/
mv ./histories $DATADIR/

rm -fr $DATADIR/cluster/*

cd $DATADIR/raw_models/

for model in *;
do
	if [ -d ../models/3D/"$model" ]; then
		rm -r ../models/3D/"$model"
	fi
	mkdir ../models/3D/"$model"
	mv -vn "$model" ../models/3D/"$model"/model
	mv -vn ../histories/"$model".json ../models/3D/"$model"/history.json
done

rm -fr $DATADIR/raw_models
rm -fr $DATADIR/histories

cd $SCRATCHDIR
(echo "/data"; echo $DATADIR/models; echo "NEW") | singularity exec -B $SCRATCHDIR:/scratchdir -B $DATADIR/data:/data --nv /cvmfs/singularity.metacentrum.cz/NGC/TensorFlow\:23.05-tf2-py3.SIF python /scratchdir/postprocess.py

clean_scratch