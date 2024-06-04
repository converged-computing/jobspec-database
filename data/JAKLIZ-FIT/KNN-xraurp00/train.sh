#!/usr/bin/env bash
#PBS -N OCR_Train
#PBS -q gpu
#PBS -l select=1:ncpus=4:mem=22gb:scratch_local=50gb:ngpus=1:gpu_cap=cuda60:cuda_version=11.0:gpu_mem=10gb
#PBS -l walltime=2:00:00
#PBS -m ae
# TODO - change walltime to match full dataset training length
export USERNAME='xraurp00'
export PROJ_SRC="/storage/brno2/home/$USERNAME/knn/KNN-xraurp00"
export MODELS="/storage/brno2/home/$USERNAME/knn/models"
export DATA="/storage/brno2/home/$USERNAME/knn/dataset/bentham_self-supervised"

export SRC_MODEL='trocr-base-stage1'
export OUT_MODEL='base-stage1-supervised_test'

# add env module
echo "Adding py-virtualenv module."
module add py-virtualenv
# change tmp dir
export TMPDIR=$SCRATCHDIR
# create env
echo "Creating virtual environment."
python3 -m venv $SCRATCHDIR/venv
. $SCRATCHDIR/venv/bin/activate
# install requirements
echo "Installing dependencies."
pip install -U pip
pip install -r $PROJ_SRC/requirements.txt
# clean pip cache
pip cache purge
# copy ds
echo "Creating copy of required files in the scratch dir."
mkdir $SCRATCHDIR/ds
cp -r $DATA/lines_40.lmdb $SCRATCHDIR/ds
cp -r $DATA/lines.100.trn $SCRATCHDIR/ds
cp -r $DATA/lines.100.val $SCRATCHDIR/ds
cp -r $DATA/lines.trn $SCRATCHDIR/ds
cp -r $DATA/lines.val $SCRATCHDIR/ds
# copy sw
mkdir $SCRATCHDIR/src
cp -r $PROJ_SRC $SCRATCHDIR/src
# run training
echo "Running the trocr_train.py script."
cd $SCRATCHDIR/src/KNN-xraurp00
# Collect the statistics
/usr/bin/time -f 'Memory: %MKB, CPU: %P%%, Time: %E' \
python trocr_train.py \
    -m $MODELS/$SRC_MODEL \
    -t $SCRATCHDIR/ds/lines_40.lmdb \
    -l $SCRATCHDIR/ds/lines.100.trn \
    -c $SCRATCHDIR/ds/lines.100.val \
    -e 3 \
    -b 20 \
    -s $MODELS/$OUT_MODEL \
    -g
# TODO - change num. epochs to 10, batch size 20
# clean scratch
echo "Cleaning up."
cd ~
deactivate
rm -rf $SCRATCHDIR/*
# unload module
module purge
echo "Batch finished."
