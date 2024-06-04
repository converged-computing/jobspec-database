#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=15:00:00
#PBS -l mem=20GB
#PBS -N py1
#PBS -M $USER@nyu.edu
#PBS -j oe

module unload python/intel/2.7.9
module load scikit-learn/intel/0.16.1

cd HPC/kmeans/
touch output/np1_times1.txt

python np-kmeans.py -f blobs_10000_10_k20.csv -k 5 -t 100 >> output/np1_times1.txt
python np-kmeans.py -f blobs_10000_10_k20.csv -k 10 -t 100 >> output/np1_times1.txt
python np-kmeans.py -f blobs_10000_10_k20.csv -k 15 -t 100 >> output/np1_times1.txt
python np-kmeans.py -f blobs_10000_10_k20.csv -k 20 -t 100 >> output/np1_times1.txt
python np-kmeans.py -f blobs_10000_10_k20.csv -k 25 -t 100 >> output/np1_times1.txt

