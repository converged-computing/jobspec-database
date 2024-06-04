#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=1:mem=10gb:scratch_local=10gb:gpu_cap=cuda70:cl_adan=True
#PBS -l walltime=01:00:00 -q gpu

# modify/delete the above given guidelines according to your job's needs
# Please note that only one select= argument is allowed at a time.

# You can check available parameters and machines here:
# https://metavo.metacentrum.cz/pbsmon2/qsub_pbspro

# cl_konos=False:cl_gram=False

# # PBS -l select=1:ncpus=1:mem=1gb:scratch_local=4gb

# add to qsub with:
# qsub qsub_detectron2_tutorial_quickstart.sh
DATETIME=`date +"%y%m%d_%H%M%S"`

# nastaveni domovskeho adresare, v promenne $LOGNAME je ulozeno vase prihlasovaci jmeno
PROJECTDIR="/storage/plzen1/home/$LOGNAME/projects/piglegsurgery/piglegcv/incision"
DATADIR="/auto/plzen4-ntis/projects/korpusy_cv/pigleg_surgery/incision_october2021"
OUTPUTDIR="/auto/plzen4-ntis/projects/korpusy_cv/pigleg_surgery/incision_october2021_output/$DATETIME"
# SIF_IMAGE=/auto/plzen4-ntis/projects/cv/CarnivoreID/carnivore_id_v22_12_0.sif
SIF_IMAGE=/auto/plzen4-ntis/projects/cv/CarnivoreID/carnivore_id_v22_05.sif


echo "job: $PBS_JOBID running on: `uname -n`"



# nastaveni automatickeho vymazani adresare SCRATCH pro pripad chyby pri behu ulohy
trap 'clean_scratch' TERM EXIT

# vstup do adresare SCRATCH, nebo v pripade neuspechu ukonceni s chybovou hodnotou rovnou 1
cd $SCRATCHDIR || exit 1

# priprava vstupnich dat (kopirovani dat na vypocetni uzel)
mkdir -p $SCRATCHDIR/data/orig
mkdir -p $SCRATCHDIR/data/processed/
mkdir -p $OUTPUTDIR
cp -r $DATADIR/* $SCRATCHDIR/data/orig/

echo "DATADIR=$DATADIR"
echo "ls DATADIR :"
ls $DATADIR

# list all files in SCRATCHDIR/data/orig
echo "SCRATCHDIR=$SCRATCHDIR"
echo "find SCRATCHDIR/data/orig/ :"
find $SCRATCHDIR/data/orig/

# activate environment option 1: miniconda installed
#module add cuda-10.1
#module add conda-modules-py37
#module add gcc-8.3.0

#source conda activate drawnUI-conda
#conda activate /storage/plzen1/home/$LOGNAME/.conda/envs/drawnUI-conda

# We download the pre-trained checkpoints for inference and finetuning.
echo "pwd: `pwd`"
mkdir -p $SCRATCHDIR/checkpoints

CHECKPOINT_PTH=faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth

if [ ! -f $SCRATCHDIR/checkpoints/$CHECKPOINT_PTH ]; then
  wget --progress=bar:force:noscroll -c https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/$CHECKPOINT_PTH \
      -O $SCRATCHDIR/checkpoints/$CHECKPOINT_PTH
fi

#export PATH=/storage/plzen1/home/$LOGNAME/miniconda3/bin:$PATH
#source activate mytorch

# this is because of python click
export LC_ALL=C.UTF-8
export LANG=C.UTF-8


# spusteni aplikace - samotny vypocet
# Put your code here
# SIF_IMAGE=/auto/plzen4-ntis/projects/cv/CarnivoreID/carnivore_id\:v1.0.sif
singularity exec -B $SCRATCHDIR:$SCRATCHDIR $SIF_IMAGE  python  $PROJECTDIR/mmdetection_custom_dataset_detection.py
# singularity exec -B $SCRATCHDIR:$SCRATCHDIR /auto/plzen4-ntis/projects/cv/CarnivoreID/carnivore_id\:v1.0.sif python  $PROJECTDIR/mmdetection_custom_dataset_detection.py

# kopirovani vystupnich dat z vypocetnicho uzlu do domovskeho adresare,
# pokud by pri kopirovani doslo k chybe, nebude adresar SCRATCH vymazan pro moznost rucniho vyzvednuti dat
cp -r $SCRATCHDIR/data/processed/* $OUTPUTDIR || export CLEAN_SCRATCH=false
#cp results.txt $OUTPUTDIR || export CLEAN_SCRATCH=false
find $OUTPUTDIR
