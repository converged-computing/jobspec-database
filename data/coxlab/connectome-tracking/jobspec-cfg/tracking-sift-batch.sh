#!/bin/sh

dir=~/Documents/Research/connectome-tracking

# enter directory
cd ${dir}

# make C files and run klt-track3
make
./klt-track3

# run extractFeatures
echo 'python extractFeatures.py --feat klt'
python extractFeatures.py --feat klt

# run evalTrackedPoints 
./matlab_batch.sh evalTrackedPoints klt

