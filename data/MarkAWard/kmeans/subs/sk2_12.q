#!/bin/bash
#PBS -l nodes=1:ppn=12
#PBS -l walltime=15:00:00
#PBS -l mem=20GB
#PBS -N sk2_12
#PBS -M $USER@nyu.edu
#PBS -j oe

module unload python/intel/2.7.9
module load scikit-learn/intel/0.16.1

cd HPC/kmeans/
touch output/sk2_times12.txt

./time_sklearn2.sh 12 >> output/sk2_times12.txt
