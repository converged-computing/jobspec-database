#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l walltime=5:00:00
#PBS -l mem=20GB
#PBS -N sk1_4
#PBS -M $USER@nyu.edu
#PBS -j oe

module unload python/intel/2.7.9
module load scikit-learn/intel/0.16.1

cd HPC/kmeans/
touch output/sk1_times4.txt

./time_sklearn1.sh 4 >> output/sk1_times4.txt
