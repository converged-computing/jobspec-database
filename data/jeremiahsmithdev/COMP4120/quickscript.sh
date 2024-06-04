#!/bin/bash

# Created by c3238179, last update: 28/05/19.

#PBS -l select=1:ncpus=16:mem=32GB
#PBS -l walltime=24:00:00           
#PBS -k oe
#PBS -M c3238179@uon.edu.au

source /etc/profile.d/modules.sh

module load python/3.6.3
module load tensorflow/1.7.0-python3.6
module load keras/2.2.2-python3.6
module load scikit-learn/1.19.1-python3.6
module load nltk/3.3.0-python3.6
module load scipy/1.2.1-python3.6
module load deap/1.2.2-python3.6

cd $PBS_O_WORKDIR

python3 GA_Darwin.py >> /home/c3238179/output

