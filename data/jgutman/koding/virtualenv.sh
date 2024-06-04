#!/bin/bash

#PBS -l mem=4GB
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=8
#PBS -M jg3862@nyu.edu
#PBS -m ae

module load python/intel/2.7.6
module load virtualenv/12.1.1
module load scikit-learn/intel/0.17

cd $SCRATCH/koding
virtualenv venv
source venv/bin/activate

pip install --upgrade pip --trusted-host None
pip install wheel --trusted-host None
pip wheel --wheel-dir=./wheelhouse pandas==0.17.0	
pip install --upgrade --no-index --use-wheel --force-reinstall --find-links=./wheelhouse pandas==0.17.0
pip wheel --wheel-dir=./wheelhouse gensim==0.12.3	
pip install --upgrade --no-index --use-wheel --force-reinstall --find-links=./wheelhouse gensim==0.12.2

pip freeze > requirements.txt
deactivate

source venv/bin/activate
pip install --trusted-host None --no-index --use-wheel --upgrade -r requirements.txt

pip show nltk
pip show scipy
pip show numpy
pip show gensim
pip show sklearn
