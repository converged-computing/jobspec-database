#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=1:mem=10gb:cl_konos=False:cl_gram=False:scratch_local=10gb
#PBS -l walltime=01:00:00 -q gpu
# modify/delete the above given guidelines according to your job's needs
# Please note that only one select= argument is allowed at a time.

# # PBS -l select=1:ncpus=1:mem=1gb:scratch_local=4gb

# add to qsub with:
# qsub qsub_pyt_tutorial_quickstart.sh

# nastaveni domovskeho adresare, v promenne $LOGNAME je ulozeno vase prihlasovaci jmeno
LOGDIR="/storage/plzen1/home/$LOGNAME/projects/tutorials/metacentrum/tensorflow_quickstart/"
PROJECTDIR="/storage/plzen1/home/$LOGNAME/projects/tutorials/metacentrum/tensorflow_quickstart/"
#DATADIR="/storage/plzen4-ntis/projects/queetech/medical/processed/scaffan2019/metacentrum"


echo "job: $PBS_JOBID running on: `uname -n`"


# nastaveni automatickeho vymazani adresare SCRATCH pro pripad chyby pri behu ulohy
trap 'clean_scratch' TERM EXIT

# vstup do adresare SCRATCH, nebo v pripade neuspechu ukonceni s chybovou hodnotou rovnou 1
cd $SCRATCHDIR || exit 1

# priprava vstupnich dat (kopirovani dat na vypocetni uzel)
# cp $DATADIR/gaussian_test.com $SCRATCHDIR

# spusteni aplikace - samotny vypocet

# activate environment option 1: miniconda installed
export PATH=/storage/plzen1/home/$LOGNAME/miniconda3/bin:$PATH
source activate mytf


# this is because of python click
export LC_ALL=C.UTF-8
export LANG=C.UTF-8


# Put your code here
python  $PROJECTDIR/example_mnist.py > results.txt

ls
# kopirovani vystupnich dat z vypocetnicho uzlu do domovskeho adresare,
# pokud by pri kopirovani doslo k chybe, nebude adresar SCRATCH vymazan pro moznost rucniho vyzvednuti dat
cp results.txt $LOGDIR || export CLEAN_SCRATCH=false

