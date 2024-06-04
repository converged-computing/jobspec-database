#!/bin/bash
#PBS -N Keras_tuner_job
#PBS -l select=1:ncpus=1:mem=16gb:scratch_local=32gb:cluster=halmir

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR=/auto/brno2/home/nierja/Tox/src/DL
LOGDIR=/auto/brno2/home/nierja/Tox/results/logs
MODELDIR=/auto/brno2/home/nierja/Tox/results/models

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually 
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

cd $DATADIR
export TMPDIR=$SCRATCHDIR

module add python/3.8.0-gcc-rab6t cuda/cuda-11.2.0-intel-19.0.4-tn4edsz cudnn/cudnn-8.1.0.77-11.2-linux-x64-intel-19.0.4-wx22b5t
python3 -m venv HP_SEARCH2
HP_SEARCH2/bin/pip install --no-cache-dir --upgrade pip setuptools
HP_SEARCH2/bin/pip install --no-cache-dir tensorflow==2.8.0 tensorflow-addons==0.16.1 tensorflow-probability==0.16.0 tensorflow-hub==0.12.0 scipy numpy pandas tabulate matplotlib rdkit talos mordred matplotlib molvs keras_tuner scikit-learn==1.1.2
source ./HP_SEARCH2/bin/activate
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# run the hyperparameter search, while logging into the $SCRATCHDIR
python Tox21_tuner.py --target=$target --fp=$fp --pca=$pca --weighted=$weight --ensamble=$ensamble --log_dir=$SCRATCHDIR > /dev/null 2>&1

# append the simulation results into the working directory
cat $SCRATCHDIR/DL_$target.csv >> $LOGDIR/DL_$target.csv

# store best models
cp -r $SCRATCHDIR/ensemble_model $MODELDIR
cp -r $SCRATCHDIR/best_single_model_* $MODELDIR

# clean the SCRATCH directory
clean_scratch
