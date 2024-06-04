#!/bin/bash
# VARIABLES PASSED TO THIS SCRIPT
#   - SEED ... seed to work with
#   - CMD  ... command to be run
#   - REPOSITORY ... path to your repository that will be copied
#   - PROGRAM_PATH ... path to the program in your repository (now from the VM)
#                      that will be run
#   - ENSEMBLE ... indicating whether we are creating an ensemble or not

#PBS -N speech_recognition_64bs_30e_128rnn_4l_64beam
#PBS -l select=1:mem=10gb:scratch_local=5gb:ncpus=4:ngpus=1:cuda_version=11.2:gpu_cap=cuda70
#PBS -q gpu
#PBS -l walltime=12:00:00
#PBS -M janmadera97@gmail.com
#PBS -m abe

#qstat 7524129.meta-pbs.metacentrum.cz

# storage is shared via NFSv4
DATADIR="/storage/liberec3-tul/home/maderaja"
REPOSITORY="09"
PROGRAM_PATH=""
CMD="python3 speech_recognition.py"

# test if scratch directory is set
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# clean the SCRATCH when job finishes (and data 
# are successfully copied out) or is killed
trap 'clean_scratch' TERM EXIT

# Prepare scratch directory for singularity
chmod 700 $SCRATCHDIR
mkdir $SCRATCHDIR/tmp
export SINGULARITY_TMPDIR=$SCRATCHDIR/tmp

# Prepare NGC container
ls /cvmfs/singularity.metacentrum.cz

# Copy repository with code to be run
cp -r $DATADIR/$REPOSITORY $SCRATCHDIR || { echo >&2 "Error while copying input file(s)!"; exit 2; }

cd $SCRATCHDIR

# Create a script that will be run in a NGC container run by Singularity
echo "cd $SCRATCHDIR/$REPOSITORY/$PROGRAM_PATH" > my_new_script.sh
echo "$CMD > out" >> my_new_script.sh

# --nv for gpu, bind scratch directory
singularity exec --bind $SCRATCHDIR --nv /cvmfs/singularity.metacentrum.cz/NGC/TensorFlow\:21.02-tf2-py3.SIF bash my_new_script.sh

# print what command has been run and print the output of the program
echo "$CMD"
cd "$SCRATCHDIR/$REPOSITORY/$PROGRAM_PATH"

# copy resources from scratch directory back on disk
# field, if not successful, scratch is not deleted
cp -r logs $DATADIR/$REPOSITORY || echo >&2 "Export of data was unsuccessful!"
cp out $DATADIR/$REPOSITORY || echo >&2 "Export of data was unsuccessful!"

clean_scratch