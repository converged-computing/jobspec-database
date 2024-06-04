#!/bin/sh
#PBS -q default
#PBS -l walltime=72:00:00
#PBS -m abe
#PBS -M osb-tkyk@g.ecc.u-tokyo.ac.jp

### must include these lines  
test $PBS_O_WORKDIR && cd $PBS_O_WORKDIR
# show the calculation node
echo `hostname`
# run the environment module
. /usr/local/Modules/init/profile.sh
### Write your qsub script from here.


# environmental setting
## setting for Renv and load specific version of R
export PATH="$HOME/.Renv/bin:$PATH"
eval "$(Renv init -)"
Renv shell 3.6.1

## language
export LANG=en_US.UTF-8

## LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/users/tosabe/local/lib:$LD_LIBRARY_PATH"
export LD_RUN_PATH="/users/tosabe/local/lib:$LD_RUN_PATH"

## PATH
export PATH="/users/tosabe/local/bin:/users/tosabe/.local/bin:$PATH"

# main
#pipenv install snakemake
pipenv run snakemake -p -j 12
