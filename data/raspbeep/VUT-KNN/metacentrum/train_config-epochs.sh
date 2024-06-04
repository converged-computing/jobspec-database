#!/bin/bash
#PBS -N f2s_CycleGAN_with_ResNet_and_feature_loss
#PBS -l walltime=12:00:00
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_local=16gb
#PBS -m ae
#PBS -j oe

# set parameters:
# number of epochs:
NUMBER_OF_EPOCHS=200
# git branch name
BRANCH_NAME=discriminator-change
# homedir:
HOMEDIR=/storage/brno2/home/$PBS_O_LOGNAME
# dataset (pick one):
DATADIR=/storage/praha1/home/xsvobo1x/datasets/faces2sketches
# DATADIR=/storage/praha1/home/xsvobo1x/datasets/selfie2anime
# DATADIR=/storage/praha1/home/xsvobo1x/datasets/horse2zebra
# DATADIR=/storage/praha1/home/xsvobo1x/datasets/cat2dog
# DATADIR=/storage/praha1/home/xsvobo1x/datasets/mushrooms
# DATADIR=$HOMEDIR/data/  # own dataset
# info: DATADIR must contain a folder with dataset:
# data/
#   train/
#     class1/
#     class2/
#   val/
#     class1/
#     class2/

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $HOMEDIR/jobs_info.txt

# loads the python modules
export PYTHONUSERBASE=/storage/brno12-cerit/home/xkrato61/pip3_libs

# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# navigate to scratch directory
cd $SCRATCHDIR

# create saved_images directory
mkdir saved_images

# clone repository with the source code
git clone https://github.com/raspbeep/VUT-KNN.git

# navigate to code directory
cd VUT-KNN/

# checkout git branch
git checkout $BRANCH_NAME

# run training
python3 train.py --save $SCRATCHDIR/saved_images --data $DATADIR --epochs $NUMBER_OF_EPOCHS > $HOMEDIR/training.out || { echo >&2 "Training ended up erroneously (with a code $?) !!"; exit 3; }

mv checkpoint* $SCRATCHDIR

# navigate to the scratch directory
cd $SCRATCHDIR

# zip the generated images
zip -r "saved_images_$PBS_JOBID.zip" saved_images checkpoint*

# move zip to the home dir
mv "saved_images_$PBS_JOBID.zip" $HOMEDIR || { echo >&2 "Result file(s) copying failed (with a code $?) !!"; exit 4; }

# clean the SCRATCH directory
clean_scratch
