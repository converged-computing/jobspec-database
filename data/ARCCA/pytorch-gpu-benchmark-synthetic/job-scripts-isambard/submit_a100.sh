#!/bin/bash
#PBS -N pytorch-bench
#PBS -q ampereq
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=12:00:00

set -eu

module purge
module load cray-python/3.9.4.2
module list

cd $PBS_O_WORKDIR

#source venv-torch1121/bin/activate
source venv-torch1121-cuda102/bin/activate

count=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
echo count $count
count=1

echo 'start'
for (( c=$count; c>=1; c-- ))
do
      python3 benchmark_models.py -g $c&& &>/dev/null
done
echo 'end'
