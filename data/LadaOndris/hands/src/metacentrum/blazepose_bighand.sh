#!/bin/bash
#PBS -N BLAZEPOSE-BIGHAND
#PBS -q gpu
#PBS -l select=1:ncpus=32:ngpus=1:mem=64gb:cpu_flag=avx512dq:scratch_ssd=50gb:gpu_cap=cuda75:cl_adan=True
#PBS -l walltime=24:00:00
#PBS -m abe

DATADIR=/storage/brno6/home/ladislav_ondris/IBT
SCRATCHDIR="$SCRATCHDIR/IBT"
mkdir $SCRATCHDIR

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

module add conda-modules-py37
conda env remove -n ibt_bighand
conda create -n ibt_bighand python=3.7
conda activate ibt_bighand
conda install matplotlib
conda install tensorflow-gpu
conda install scikit-learn
conda install scikit-image
pip install opencv-python
pip install numpy==1.19.5
pip install gast==0.3.3
pip install tensorflow-addons
pip install tensorflow-probability==0.12.1
conda list

# Copy source code
cp -r "$DATADIR/src" "$SCRATCHDIR/" || { echo >&2 "Couldnt copy srcdir to scratchdir."; exit 2; }

# Prepare datasets folder
mkdir "$SCRATCHDIR/datasets"
mkdir "$SCRATCHDIR/datasets/bighand"
# Copy full_annotations.tar
cp -r "$DATADIR/datasets/bighand/full_annotation.tar" "$SCRATCHDIR/datasets" || { echo >&2 "Couldnt copy full_annotations.tar"; exit 2; }
# Extract it
tar -xf "$SCRATCHDIR/datasets/full_annotation.tar" -C "$SCRATCHDIR/datasets/bighand" || { echo >&2 "Couldnt extract full_annotations.tar"; exit 2; }

# Copy Subject_1.tar
cp -r "$DATADIR/datasets/bighand/Subject_1.tar" "$SCRATCHDIR/datasets" || { echo >&2 "Couldnt copy Subject_1.tar"; exit 2; }
# Extract it
tar -xf "$SCRATCHDIR/datasets/Subject_1.tar" -C "$SCRATCHDIR/datasets/bighand" || { echo >&2 "Couldnt extract Subject_1.tar"; exit 2; }

# Copy Subject_2.tar
cp -r "$DATADIR/datasets/bighand/Subject_2.tar" "$SCRATCHDIR/datasets" || { echo >&2 "Couldnt copy Subject_2.tar"; exit 2; }
# Extract it
tar -xf "$SCRATCHDIR/datasets/Subject_2.tar" -C "$SCRATCHDIR/datasets/bighand" || { echo >&2 "Couldnt extract Subject_2.tar"; exit 2; }

# Copy Subject_4.tar
cp -r "$DATADIR/datasets/bighand/Subject_4.tar" "$SCRATCHDIR/datasets" || { echo >&2 "Couldnt copy Subject_4.tar"; exit 2; }
tar -xf "$SCRATCHDIR/datasets/Subject_4.tar" -C "$SCRATCHDIR/datasets/bighand" || { echo >&2 "Couldnt extract Subject_4.tar"; exit 2; }

# Copy Subject_5.tar
cp -r "$DATADIR/datasets/bighand/Subject_5.tar" "$SCRATCHDIR/datasets" || { echo >&2 "Couldnt copy Subject_5.tar"; exit 2; }
tar -xf "$SCRATCHDIR/datasets/Subject_5.tar" -C "$SCRATCHDIR/datasets/bighand" || { echo >&2 "Couldnt extract Subject_5.tar"; exit 2; }

# Copy Subject_6.tar
cp -r "$DATADIR/datasets/bighand/Subject_6.tar" "$SCRATCHDIR/datasets" || { echo >&2 "Couldnt copy Subject_6.tar"; exit 2; }
tar -xf "$SCRATCHDIR/datasets/Subject_6.tar" -C "$SCRATCHDIR/datasets/bighand" || { echo >&2 "Couldnt extract Subject_6.tar"; exit 2; }

# Copy Subject_7.tar
cp -r "$DATADIR/datasets/bighand/Subject_7.tar" "$SCRATCHDIR/datasets" || { echo >&2 "Couldnt copy Subject_7.tar"; exit 2; }
tar -xf "$SCRATCHDIR/datasets/Subject_7.tar" -C "$SCRATCHDIR/datasets/bighand" || { echo >&2 "Couldnt extract Subject_7.tar"; exit 2; }

cp -r "$DATADIR/saved_models" "$SCRATCHDIR/" || { echo >&2 "Couldnt copy saved models"; exit 2; }

export PYTHONPATH=$SCRATCHDIR
python3 $SCRATCHDIR/src/estimation/blazepose/trainers/trainer.py --config config_blazepose_regress.json --verbose 0 --batch-size 64 --weights "$SCRATCHDIR/saved_models/20211103-011843/weights.40.h5"

cp -r $SCRATCHDIR/logs $DATADIR/ || { echo >&2 "Couldnt copy logs to datadir."; exit 3; }
cp -r $SCRATCHDIR/saved_models $DATADIR/ || { echo >&2 "Couldnt copy saved_models to datadir."; exit 3; }
clean_scratch
