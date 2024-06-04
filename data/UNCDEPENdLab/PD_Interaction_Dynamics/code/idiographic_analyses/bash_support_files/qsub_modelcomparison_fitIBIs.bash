#!/usr/bin/env sh

#PBS -l walltime=4320:00:00
#PBS -l nodes=1:ppn=37
#PBS -A mnh5174_collab
#PBS -j oe
# mnh5174@psu.edu
#PBS -M ams939@psu.edu
#PBS -m abe

env
cd $PBS_O_WORKDIR

module load matlab
export matlab_cpus=37

echo "Currently in: $PWD"

matlab -nodisplay -r modelcomparison_fitIBIs
